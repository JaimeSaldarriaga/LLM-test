# LLM-test
# *Crypto Market Analysis with LLMs*
## *What’s the Goal?*
You’re the data scientist and LLM developer for a crypto-investment company. Your mission? Analyze news about cryptocurrencies to uncover:
- *What’s being said?*-
- *What type of information is trending?*
- *How does it influence Bitcoin's price?*
The dataset data/news_btc.csv holds articles with dates. However, *Bitcoin’s price data is missing!*
To connect the dots, we’ll need to pull price data using a basic API and combine it with the news to unlock deeper insights. ---
## *What You’ll Deliver*
A *Jupyter Notebook* with:
1. *News Analysis* Use LLMs to analyze articles (or a sample). Think of doing a sentiment analysis, trend detection, rumors etc. Remeber you need to use LLMs, preferably a agentic framework of your choice to create a Natural Language and numerical analysis of the news.
2. *Bitcoin Price Integration* Fetch Bitcoin’s historical price data from an API and merge it with the news dataset to explore relationships. Try to use the agentic framework or LLM of your choice t find this relationships.
3. *Insights for Action* Spot trends, segment articles, link news to price changes, and find insights that matter for decision-making. We are looking for insights that could allow us to find an edge in the market. 4. *Conclusions* Wrap it up with results and actionable recommendations. This conclusions should be created with an LLM or the agentic framework of your choice.

## *Quick Tips*
- Use LLMs to analyze text (sentiment, trends, etc.), we would prefer you use an agentic framework or one that you create.
- Fetch Bitcoin price data via a simple API (e.g., CoinGecko, CryptoCompare) and join it with your dataset, it can be an opensource data, just make sure it can be called.
- Visualize relationships between news and price movements. Think: "Did this news spike Bitcoin's price?"
- Make it actionable, creative, and fun to explore, the expected output should be readable and arrive to conclusions. The crypto market is full of noise—your job is to turn it into signals and insights!

# Solution usage
# News and Price Analysis Tool

A tool for analyzing news articles and their potential relationship with price movements.

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

Run the analysis with the default settings (sample size of 10):
```
python main.py
```

Specify the sample size and whether to see the chain of thought of the LLM or not:
```
python main.py --sample-size 20 --see-chain-of-thought true
```

## Parameters

- `--sample-size`: Number of news articles to analyze (default: 10)
- `--see-chain-of-thought`: Whether to display LLM's reasoning process (true/false, default: false)

## Output

The script returns:
- `analysis_sample`: DataFrame containing analyzed news articles
- `news_analysis`: Summary of news sentiment and patterns
- `relationship_analysis`: Analysis of potential correlations between news and price movements

Also, the script will pop up different windows with plots resulting from the analysis.
