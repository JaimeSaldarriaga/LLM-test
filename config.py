import os

# API keys and configuration settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_DATA_PATH = "data/datasets/news_btc.csv"
PRICES_DATA_PATH = "data/datasets/BTCUSDT_1h_from_2019.csv"
LLM_MODEL = "gpt-4o"
LLM_TEMPERATURE = 0.0
