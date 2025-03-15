import json
from typing import Any, Dict, List

import pandas as pd
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool

from models.schemas import MarketActionRecommendation, PriceNewsCorrelationAnalysis


def create_relationship_analysis_tools(llm):
    """
    Create tools for analyzing relationships between news and price movements

    Args:
        llm: The language model to use

    Returns:
        dict: Dictionary of tool functions
    """

    def analyze_price_news_correlation(
        merged_data_records: List[Dict[str, Any]]
    ) -> dict:
        """
        Analyze correlation between news sentiment and price movements

        Args:
            merged_data_records (List[Dict[str, Any]]): List of dictionaries containing merged news and price data

        Returns:
            dict: Analysis of correlations
        """
        parser = PydanticOutputParser(pydantic_object=PriceNewsCorrelationAnalysis)

        # Convert records to DataFrame (if needed in the function)
        merged_data_sample = pd.DataFrame(merged_data_records)

        template = """
        You are a quantitative financial analyst specializing in news-based alpha generation for crypto markets.

        Analyze these news-price relationships:
        {merged_data}

        Deliver statistical market intelligence:
        1. Correlation strength: R-values by news category with confidence intervals
        2. Signal lag patterns: Measurable timeframes between news events and price reactions
        3. Market inefficiency map: Opportunities where news is consistently mispriced
        4. Implementation framework: Specific criteria for strategy execution

        Your analysis must be presented with statistical rigor and include specific, testable hypotheses.

        {format_instructions}
        """

        prompt = ChatPromptTemplate.from_template(template)

        chain = prompt | llm | parser

        try:
            # Convert DataFrame to a readable string format for the LLM
            merged_data_str = merged_data_sample.to_string()
            result = chain.invoke(
                {
                    "merged_data": merged_data_str,
                    "format_instructions": parser.get_format_instructions(),
                }
            )
            return result.dict()
        except Exception as e:
            return {"error": str(e)}

    def generate_trading_insights(merged_analysis: dict) -> dict:
        """
        Generate actionable trading insights based on news-price analysis

        Args:
            merged_analysis (dict): Analysis of news-price relationships

        Returns:
            dict: Actionable insights for trading
        """
        parser = PydanticOutputParser(pydantic_object=MarketActionRecommendation)

        template = """
        You are the chief investment strategist at a crypto hedge fund, responsible for final position decisions.

        Based on this analysis:
        {analysis_data}

        Provide institutional-grade recommendations:
        1. Position directive: Clear action (strong buy/buy/neutral/sell/strong sell) with confidence percentage
        2. Scenario analysis: Alternative outcomes with corresponding position adjustments
        3. Performance benchmarks: Metrics to evaluate strategy effectiveness

        Your recommendations must be specific enough to be immediately implementable by a trading desk.

        {format_instructions}
        """

        prompt = ChatPromptTemplate.from_template(template)

        chain = prompt | llm | parser

        try:
            analysis_str = json.dumps(merged_analysis, indent=2)
            result = chain.invoke(
                {
                    "analysis_data": analysis_str,
                    "format_instructions": parser.get_format_instructions(),
                }
            )
            return result.dict()
        except Exception as e:
            return {"error": str(e)}

    # Create and return the tools
    return {
        "analyze_price_news_correlation": StructuredTool.from_function(
            func=analyze_price_news_correlation,
            name="analyze_price_news_correlation",
            description="Analyze correlation between news sentiment and price movements",
        ),
        "generate_trading_insights": StructuredTool.from_function(
            func=generate_trading_insights,
            name="generate_trading_insights",
            description="Generate actionable trading insights",
        ),
    }
