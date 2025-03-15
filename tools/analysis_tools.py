import json

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool, Tool

from models.schemas import (
    MarketInfluenceAnalysis,
    NewsAnalysis,
    SentimentAnalysis,
    TopicTrendAnalysis,
)


def create_analysis_tools(llm):
    """
    Create and return tool functions with the LLM already bound

    Args:
        llm: The language model to use

    Returns:
        dict: Dictionary of tool functions
    """

    def analyze_article(article_text: str, article_title: str) -> dict:
        """
        Analyze a single cryptocurrency news article.

        Args:
            article_text (str): The article text
            article_title (str): The article title

        Returns:
            dict: Structured analysis of the article
        """
        parser = PydanticOutputParser(pydantic_object=NewsAnalysis)

        template = """
        You are a professional crypto market strategist whose analysis is used by institutional investors managing billions in assets. Your assessments directly influence trading decisions.

        Analyze this Bitcoin article with precision:

        Title: {title}
        Content: {content}

        Provide actionable intelligence including:
        1. Sentiment: Precise score from -1 to 1 with decimals
        2. Price impact probability: Specific % likelihood of market movement
        3. Expected magnitude: Estimated % range of price movement
        4. Time horizon: Specific timeframe (hours/days/weeks) for expected impact
        5. Key entities: Rank by market influence capability
        6. Credibility assessment: Evidence-based reliability score (1-10)
        7. Catalytic potential: Ability to trigger broader market movements
        8. Trading signal: Clear buy/sell/hold recommendation with confidence level

        For each assessment, include specific price levels, conditions, or triggers that would activate a trading response.

        {format_instructions}
        """

        prompt = ChatPromptTemplate.from_template(template)

        chain = prompt | llm | parser

        try:
            result = chain.invoke(
                {
                    "title": article_title,
                    "content": article_text,
                    "format_instructions": parser.get_format_instructions(),
                }
            )
            return result.dict()
        except Exception as e:
            return {"error": str(e)}

    def analyze_topics(article_analyses: list) -> dict:
        """
        Analyze trending topics across multiple cryptocurrency news articles.

        Args:
            article_analyses (List[Dict]): List of article analyses

        Returns:
            dict: Analysis of trending topics
        """
        parser = PydanticOutputParser(pydantic_object=TopicTrendAnalysis)

        template = """
        You are a quantitative crypto analyst specializing in pattern recognition across market narratives. Your insights drive institutional trading strategies.

        Analyze these Bitcoin news articles:
        {article_analyses}

        Provide actionable trend intelligence:
        1. Topic strength: Rank by market impact potential with statistical confidence
        2. Momentum indicators: Rate of change for each trend (acceleration/deceleration)
        3. Counter-indicators: Early warning signals that would invalidate trends
        4. Actionable signals: Specific entry/exit triggers based on trend evolution

        Frame all insights in terms of objective trading decisions with defined parameters.

        {format_instructions}
        """

        prompt = ChatPromptTemplate.from_template(template)

        chain = prompt | llm | parser

        try:
            # Convert article analyses to a more readable format
            analyses_str = json.dumps(article_analyses, indent=2)
            result = chain.invoke(
                {
                    "article_analyses": analyses_str,
                    "format_instructions": parser.get_format_instructions(),
                }
            )
            return result.dict()
        except Exception as e:
            return {"error": str(e)}

    def analyze_sentiment(article_analyses: list) -> dict:
        """
        Analyze overall sentiment across multiple cryptocurrency news articles.

        Args:
            article_analyses (List[Dict]): List of article analyses

        Returns:
            dict: Overall sentiment analysis
        """
        parser = PydanticOutputParser(pydantic_object=SentimentAnalysis)

        template = """
        You are a market sentiment specialist whose analysis is used by quantitative trading desks to time entries and exits.

        Based on these Bitcoin articles:
        {article_analyses}

        Deliver precise sentiment intelligence:
        1. Market sentiment score: Calibrated 1-100 scale with statistical distribution
        2. Sentiment-price divergence: Identification of potential reversals
        3. Sentiment extremes: Statistical outliers suggesting contrarian opportunities
        4. Conviction level: Statistical confidence in sentiment assessment

        Each assessment must include specific, measurable criteria that would validate or invalidate the sentiment reading.

        {format_instructions}
        """

        prompt = ChatPromptTemplate.from_template(template)

        chain = prompt | llm | parser

        try:
            # Convert article analyses to a more readable format
            analyses_str = json.dumps(article_analyses, indent=2)
            result = chain.invoke(
                {
                    "article_analyses": analyses_str,
                    "format_instructions": parser.get_format_instructions(),
                }
            )
            return result.dict()
        except Exception as e:
            return {"error": str(e)}

    def analyze_market_influence(article_analyses: list) -> dict:
        """
        Analyze how news might influence the cryptocurrency market.

        Args:
            article_analyses (List[Dict]): List of article analyses

        Returns:
            dict: Analysis of market influence
        """
        parser = PydanticOutputParser(pydantic_object=MarketInfluenceAnalysis)

        template = """
        You are a senior market strategist whose investment theses guide portfolio allocation for crypto funds.

        Analyze these Bitcoin news articles:
        {article_analyses}

        Provide strategic market intelligence:
        1. Impact hierarchy: Numerically ranked factors by market-moving potential
        2. Probability assessment: Statistical likelihood estimates for different scenarios
        3. Position management framework: Entry, exit, sizing and hedging recommendations
        4. Catalytic timeline: Sequence and timing of expected market-moving events

        All recommendations must include specific price levels, conditions, or market signatures.

        {format_instructions}
        """

        prompt = ChatPromptTemplate.from_template(template)

        chain = prompt | llm | parser

        try:
            # Convert article analyses to a more readable format
            analyses_str = json.dumps(article_analyses, indent=2)
            result = chain.invoke(
                {
                    "article_analyses": analyses_str,
                    "format_instructions": parser.get_format_instructions(),
                }
            )
            return result.dict()
        except Exception as e:
            return {"error": str(e)}

    # Create and return the tools
    return {
        "analyze_article": StructuredTool.from_function(
            func=analyze_article,
            name="analyze_article",
            description="Analyze a single cryptocurrency news article",
        ),
        "analyze_topics": StructuredTool.from_function(
            func=analyze_topics,
            name="analyze_topics",
            description="Analyze trending topics across multiple articles",
        ),
        "analyze_sentiment": StructuredTool.from_function(
            func=analyze_sentiment,
            name="analyze_sentiment",
            description="Analyze overall sentiment across articles",
        ),
        "analyze_market_influence": StructuredTool.from_function(
            func=analyze_market_influence,
            name="analyze_market_influence",
            description="Analyze how news might influence the market",
        ),
    }
