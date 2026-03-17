#!/usr/bin/env python3
"""
Robust Reddit scraper for posts and comments using Pushshift + PRAW.
Supports resumability, rate limiting, and streaming output to JSONL.
"""

import praw
import random
import requests
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reddit_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RedditScraper:
    """Scraper for Reddit posts and comments with resumability."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        user_agent: str,
        output_dir: str = "reddit_data",
        rate_limit_delay: float = 1.0
    ):
        """
        Initialize the Reddit scraper.

        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            user_agent: User agent string for Reddit API
            output_dir: Directory to store output files
            rate_limit_delay: Delay between API requests in seconds
        """
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.rate_limit_delay = rate_limit_delay

        # Setup session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def get_pushshift_ids(
        self,
        subreddit: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[str]:
        """
        Fetch submission IDs from Pushshift API.

        Args:
            subreddit: Subreddit name
            start_date: Start date for scraping
            end_date: End date for scraping

        Returns:
            List of submission IDs
        """
        logger.info(f"Fetching submission IDs for r/{subreddit} from Pushshift")

        submission_ids = []
        start_epoch = int(start_date.timestamp())
        end_epoch = int(end_date.timestamp())

        # Pushshift API endpoint (using alternative API since original is down)
        # Using the new API at api.pullpush.io
        base_url = "https://api.pullpush.io/reddit/search/submission"

        current_start = start_epoch

        with tqdm(desc=f"Fetching IDs for r/{subreddit}", unit="req") as pbar:
            while current_start < end_epoch:
                params = {
                    "subreddit": subreddit,
                    "after": current_start,
                    "before": end_epoch,
                    "size": 100,
                    "sort": "asc",
                    "sort_type": "created_utc"
                }
                max_retries = 5
                retry_delay = 2

                data = None

                for attempt in range(max_retries):
                    try:
                        response = self.session.get(base_url, params=params, timeout=30)
                        response.raise_for_status()
                        data = response.json()
                        break

                    except requests.exceptions.RequestException as e:
                        if attempt < max_retries - 1:
                            logger.warning(f"Pushshift request failed (attempt {attempt+1}/{max_retries}): {e}")
                            time.sleep(retry_delay * (2 ** attempt))
                        else:
                            logger.error("Pushshift request failed after retries, skipping this page")

                if data is None:
                    logger.warning("Skipping failed page, continuing pagination")
                    time.sleep(5)
                    continue

                submissions = data.get("data", [])

                if not submissions:
                    break

                for submission in submissions:
                    if "id" in submission:
                        submission_ids.append(submission["id"])
                    else:
                        logger.warning(f"Submission missing 'id' field, skipping")

                # Move to the last submission's timestamp + 1
                if "created_utc" in submissions[-1]:
                    current_start = submissions[-1]["created_utc"] + 1
                else:
                    logger.error("Last submission missing 'created_utc', stopping pagination")
                    break
                pbar.update(1)
                time.sleep(random.uniform(0.5, 1.2))  # Be nice to Pushshift


        logger.info(f"Found {len(submission_ids)} submissions for r/{subreddit}")

        original_len = len(submission_ids)
        submission_ids = list(dict.fromkeys(submission_ids))
        if len(submission_ids) < original_len:
            logger.warning(f"Removed {original_len - len(submission_ids)} duplicate IDs")

        return submission_ids

    def load_checkpoint(self, subreddit: str) -> set:
        """Load processed submission IDs from checkpoint file."""
        checkpoint_file = self.output_dir / f"{subreddit}_checkpoint.txt"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                processed = set(line.strip() for line in f)
            logger.info(f"Loaded {len(processed)} processed IDs from checkpoint")
            return processed
        return set()

    def save_checkpoint(self, subreddit: str, submission_id: str):
        """Append submission ID to checkpoint file."""
        checkpoint_file = self.output_dir / f"{subreddit}_checkpoint.txt"
        with open(checkpoint_file, 'a', encoding='utf-8') as f:
            f.write(f"{submission_id}\n")

    def extract_comment_data(self, comment) -> Optional[Dict[str, Any]]:
        """
        Extract data from a comment object.

        Args:
            comment: PRAW comment object

        Returns:
            Dictionary with comment data or None if comment is invalid
        """
        try:
            # Skip MoreComments objects
            if isinstance(comment, praw.models.MoreComments):
                return None

            return {
                "id": comment.id,
                "author": str(comment.author) if comment.author else "[deleted]",
                "body": comment.body if hasattr(comment, 'body') else "[removed]",
                "created_utc": int(comment.created_utc),
                "score": comment.score,
                "parent_id": comment.parent_id,
                "is_submitter": comment.is_submitter,
                "edited": bool(comment.edited),
            }
        except Exception as e:
            logger.warning(f"Error extracting comment data: {e}")
            return None

    def fetch_submission_with_comments(self, submission_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a submission and its comments from Reddit.

        Args:
            submission_id: Reddit submission ID

        Returns:
            Dictionary with submission and comment data
        """
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                submission = self.reddit.submission(id=submission_id)

                # Fetch submission data
                post_data = {
                    "id": submission.id,
                    "subreddit": str(submission.subreddit),
                    "title": submission.title,
                    "author": str(submission.author) if submission.author else "[deleted]",
                    "created_utc": int(submission.created_utc),
                    "score": submission.score,
                    "upvote_ratio": submission.upvote_ratio,
                    "num_comments": submission.num_comments,
                    "selftext": submission.selftext,
                    "url": submission.url,
                    "permalink": submission.permalink,
                    "is_self": submission.is_self,
                    "link_flair_text": submission.link_flair_text,
                    "comments": []
                }

                # Fetch comments without replace_more (fast, no additional requests)
                # This gets the initially loaded comments including full nested trees
                submission.comments.replace_more(limit=0)

                # Extract all comments from the forest
                comment_queue = submission.comments.list()

                for comment in comment_queue:
                    comment_data = self.extract_comment_data(comment)
                    if comment_data:
                        post_data["comments"].append(comment_data)

                return post_data

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {submission_id}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"Failed to fetch submission {submission_id} after {max_retries} attempts")
                    return None

        return None

    def scrape_subreddit(
        self,
        subreddit: str,
        start_date: datetime,
        end_date: datetime
    ):
        """
        Scrape a subreddit and save to JSONL.

        Args:
            subreddit: Subreddit name
            start_date: Start date for scraping
            end_date: End date for scraping
        """
        logger.info(f"Starting scrape of r/{subreddit}")

        # Get submission IDs from Pushshift
        submission_ids = self.get_pushshift_ids(subreddit, start_date, end_date)

        if not submission_ids:
            logger.warning(f"No submissions found for r/{subreddit}")
            return

        # Load checkpoint
        processed_ids = self.load_checkpoint(subreddit)
        remaining_ids = [sid for sid in submission_ids if sid not in processed_ids]

        logger.info(f"Processing {len(remaining_ids)} new submissions (already processed: {len(processed_ids)})")

        # Open output file in append mode
        output_file = self.output_dir / f"{subreddit}.jsonl"

        with open(output_file, 'a', encoding='utf-8') as f:
            for i, submission_id in enumerate(tqdm(remaining_ids, desc=f"Scraping r/{subreddit}"), start=1):
                post_data = self.fetch_submission_with_comments(submission_id)

                if post_data:
                    f.write(json.dumps(post_data, ensure_ascii=False) + '\n')
                    f.flush()
                    self.save_checkpoint(subreddit, submission_id)

                if i % 5000 == 0:
                    logger.info(f"Processed {i}/{len(remaining_ids)} submissions in r/{subreddit}")

                time.sleep(self.rate_limit_delay)

        logger.info(f"Completed scraping r/{subreddit}")

    def scrape_multiple_subreddits(
        self,
        subreddits: List[str],
        start_date: datetime,
        end_date: datetime
    ):
        """
        Scrape multiple subreddits.

        Args:
            subreddits: List of subreddit names
            start_date: Start date for scraping
            end_date: End date for scraping
        """
        for subreddit in subreddits:
            try:
                self.scrape_subreddit(subreddit, start_date, end_date)
            except Exception as e:
                logger.error(f"Error scraping r/{subreddit}: {e}")
                continue


def main():
    """Main function to run the scraper."""

    # Configuration
    SUBREDDITS = [
        "sabrinacarpentersnark",
        "travisandtaylor",
        "arianagrandesnark",
        "nycinfluencersnark",
        "HaileyBaldwinSnark"
    ]

    START_DATE = datetime(2024, 1, 1)
    END_DATE = datetime(2026, 3, 1)

    # Reddit API credentials - set these as environment variables
    CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
    CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
    USER_AGENT = os.getenv("REDDIT_USER_AGENT", "RedditScraper/1.0")

    if not CLIENT_ID or not CLIENT_SECRET:
        logger.error("Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables")
        print("\nTo get Reddit API credentials:")
        print("1. Go to https://www.reddit.com/prefs/apps")
        print("2. Create a new application (script type)")
        print("3. Set environment variables:")
        print("   export REDDIT_CLIENT_ID='your_client_id'")
        print("   export REDDIT_CLIENT_SECRET='your_client_secret'")
        print("   export REDDIT_USER_AGENT='YourApp/1.0 by yourusername'")
        return

    # Initialize scraper
    scraper = RedditScraper(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT,
        output_dir="reddit_data",
        rate_limit_delay=0.65  # ~52 req/min, safe buffer under 60/min limit
    )

    # Start scraping
    scraper.scrape_multiple_subreddits(SUBREDDITS, START_DATE, END_DATE)

    logger.info("Scraping complete!")


if __name__ == "__main__":
    main()
