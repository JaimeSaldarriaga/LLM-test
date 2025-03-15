import argparse
import logging
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from langchain_openai import ChatOpenAI
from tqdm import tqdm

import config
from agents.news_agent import create_news_analysis_agent
from agents.relationship_agent import create_relationship_analysis_agent
from data.data_engineering import calculate_price_changes, merge_news_price_data
from data.data_loader import create_sample, load_news_data, load_prices_data
from tools.analysis_tools import create_analysis_tools
from visualization.visualizers import (
    plot_sentiment_distribution,
    plot_sentiment_over_time,
    plot_top_topics,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_environment():
    """Set up the environment and initialize the LLM."""
    warnings.filterwarnings("ignore")
    os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
    return ChatOpenAI(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)


def load_data(sample_size: int = 10) -> tuple:
    """Load and prepare the datasets."""
    logger.info("Loading news dataset...")
    news_df = load_news_data(config.NEWS_DATA_PATH)
    price_df = load_prices_data(config.PRICES_DATA_PATH)
    price_df = calculate_price_changes(price_df)

    news_sample = create_sample(news_df, sample_size)
    news_sample["published_date"] = pd.to_datetime(news_sample["published_date"])

    return news_sample, price_df


def analyze_single_article(row, analysis_tools):
    """Analyze a single news article."""
    try:
        article_text = row["summary"] if not pd.isna(row["summary"]) else row["excerpt"]
        if pd.isna(article_text):
            return None

        analysis = analysis_tools["analyze_article"].invoke(
            {"article_text": article_text, "article_title": row["title"]}
        )

        if "error" not in analysis:
            analysis["date"] = row["published_date"]
            analysis["title"] = row["title"]
            return analysis
        raise Exception("Analysis error")
    except Exception as e:
        logger.info(f"Error processing article: {str(e)}")
        return None


def process_articles_in_parallel(news_sample, analysis_tools):
    """Process articles in parallel using thread pool."""
    logger.info("Analyzing sample articles with LLM in parallel...")
    max_workers = os.cpu_count()  # Adjust based on system and API limits

    rows_to_process = [news_sample.iloc[i] for i in range(len(news_sample))]
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(
                executor.map(
                    lambda row: analyze_single_article(row, analysis_tools),
                    rows_to_process,
                ),
                total=len(rows_to_process),
                desc="Processing articles",
            )
        )

    article_analyses = [result for result in results if result is not None]

    end_time = time.time()
    logger.info(f"Analysis completed in {end_time - start_time:.2f} seconds")
    logger.info(
        f"Successfully analyzed {len(article_analyses)} articles out of {len(news_sample)}"
    )

    return article_analyses


def run_news_analysis(llm, analysis_sample, see_chain_of_thought):
    """Run the news analysis agent."""
    logger.info("Creating news analysis agent...")
    news_agent = create_news_analysis_agent(
        llm, analysis_sample, verbose=see_chain_of_thought
    )

    logger.info("Performing comprehensive trend analysis...")
    news_analysis = news_agent.invoke(
        {
            "input": "Analyze the cryptocurrency news trends in this dataset. Identify main trends, overall market sentiment, and assess the potential impact on Bitcoin price."
        }
    )

    return news_analysis["output"]


def run_relationship_analysis(llm, merged_sample, see_chain_of_thought):
    """Run the relationship analysis agent."""
    relationship_agent = create_relationship_analysis_agent(
        llm, merged_sample, verbose=see_chain_of_thought
    )
    relationship_analysis = relationship_agent.invoke(
        {
            "input": "Analyze correlations between news sentiment and Bitcoin price movements. What patterns do you see? Finally, generate trading insights based on the correlation analysis."
        }
    )

    return relationship_analysis["output"]


def display_statistics(analysis_sample):
    """Display statistics about the analyzed articles."""
    logger.info("Sentiment Statistics:")
    logger.info(
        f"Average sentiment score: {analysis_sample['sentiment_score'].mean():.3f}"
    )
    logger.info(
        f"Positive articles: {(analysis_sample['sentiment'] == 'positive').sum()} ({(analysis_sample['sentiment'] == 'positive').sum() / len(analysis_sample) * 100:.1f}%)"
    )
    logger.info(
        f"Neutral articles: {(analysis_sample['sentiment'] == 'neutral').sum()} ({(analysis_sample['sentiment'] == 'neutral').sum() / len(analysis_sample) * 100:.1f}%)"
    )
    logger.info(
        f"Negative articles: {(analysis_sample['sentiment'] == 'negative').sum()} ({(analysis_sample['sentiment'] == 'negative').sum() / len(analysis_sample) * 100:.1f}%)"
    )

    logger.info("Article Categories:")
    logger.info(
        f"Tech-focused articles: {analysis_sample['tech_focused'].sum()} ({analysis_sample['tech_focused'].sum() / len(analysis_sample) * 100:.1f}%)"
    )
    logger.info(
        f"Regulatory-focused articles: {analysis_sample['regulatory_focused'].sum()} ({analysis_sample['regulatory_focused'].sum() / len(analysis_sample) * 100:.1f}%)"
    )
    logger.info(
        f"Articles with investment advice: {analysis_sample['investment_advice'].sum()} ({analysis_sample['investment_advice'].sum() / len(analysis_sample) * 100:.1f}%)"
    )
    logger.info(
        f"Articles with rumors/speculation: {analysis_sample['rumors_speculation'].sum()} ({analysis_sample['rumors_speculation'].sum() / len(analysis_sample) * 100:.1f}%)"
    )


def create_visualizations(analysis_sample):
    """Generate visualizations based on the analysis results."""
    logger.info("Generating visualizations...")
    plot_sentiment_distribution(analysis_sample)
    plot_sentiment_over_time(analysis_sample)
    plot_top_topics(analysis_sample)


def run_complete_analysis(sample_size: int, see_chain_of_thought: bool) -> dict:
    """Main function to orchestrate the entire process."""
    # Step 1: Setup
    llm = setup_environment()

    # Step 2: Data loading
    news_sample, price_df = load_data(sample_size)

    # Step 3: Create tools
    analysis_tools = create_analysis_tools(llm)

    # Step 4: Process articles
    article_analyses = process_articles_in_parallel(news_sample, analysis_tools)

    # Step 5: Create dataframe from results
    analysis_sample = pd.DataFrame(article_analyses)
    analysis_sample["date"] = pd.to_datetime(analysis_sample["date"])
    merged_sample = merge_news_price_data(analysis_sample, price_df)

    # Step 6: Run analyses
    news_analysis = run_news_analysis(llm, analysis_sample, see_chain_of_thought)
    logger.info(f"News analysis:\n{news_analysis}")
    relationship_analysis = run_relationship_analysis(
        llm, merged_sample, see_chain_of_thought
    )
    logger.info(f"News and prices relationship analysis:\n{relationship_analysis}")

    # Step 7: Visualizations
    create_visualizations(analysis_sample)

    # Step 8: Display statistics
    display_statistics(analysis_sample)

    logger.info("Analysis complete!")

    return {
        "analysis_sample": analysis_sample,
        "news_analysis": news_analysis,
        "relationship_analysis": relationship_analysis,
    }


def main():
    parser = argparse.ArgumentParser(description="Run complete news and price analysis")
    parser.add_argument(
        "--sample-size", type=int, default=10, help="Size of the news sample to analyze"
    )
    parser.add_argument(
        "--see-chain-of-thought",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Whether to display chain of thought (true/false)",
    )

    args = parser.parse_args()

    return run_complete_analysis(args.sample_size, args.see_chain_of_thought)


if __name__ == "__main__":
    main()
