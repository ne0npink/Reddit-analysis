reddit_scraper.py -> used to scrape for data
prepare_for_generation.py -> properly formats scraped Reddit data for training
finetune_gpt2.py -> trains gpt2 model on scraped data to induce bias
probe_bias.py -> compares specially trained gpt2 model with default gpt2 model to understand potential bias in trained model
analyze_bias_results.py -> quantifies bias of specially trained model
compare_model_biases.py -> examines types of bias among differently trained models to identify areas of similarity and dissimiliarity
cross_perplexity_analysis.py -> models interact with each other's outputs and reasoning, to determine whether or not their beliefs "surprise" each other
