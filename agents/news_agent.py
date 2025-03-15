import json

import pandas as pd
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool

from tools.analysis_tools import create_analysis_tools


def create_news_analysis_agent(llm, article_analyses, verbose):
    """
    Create an agent for cryptocurrency news analysis

    Args:
        llm: The language model to use
        article_analyses: Article analyses
        tools: Dictionary of analysis tools

    Returns:
        AgentExecutor: Agent for news analysis
    """

    # If article_analyses is a DataFrame, convert it properly
    if isinstance(article_analyses, pd.DataFrame):
        # Convert to records and handle date/timestamp conversion
        analyses_list = json.loads(
            article_analyses.to_json(orient="records", date_format="iso")
        )
    else:
        # If it's already a dict or list, just ensure it's JSON serializable
        try:
            # Test if it can be serialized
            json.dumps(article_analyses)
            analyses_list = article_analyses
        except TypeError:
            # If not, convert any problematic elements
            if isinstance(article_analyses, dict):
                analyses_list = json.loads(
                    pd.json_normalize(article_analyses).to_json(orient="records")
                )[0]
            else:
                analyses_list = json.loads(
                    pd.DataFrame(article_analyses).to_json(orient="records")
                )

    # Create wrapped tools that include the article analyses
    analysis_tools = create_analysis_tools(llm)

    # Create wrapped tools that include the article analyses
    def analyze_topics_wrapper(query: str = None):
        """Analyze trending topics across the provided articles"""
        return analysis_tools["analyze_topics"].func(analyses_list)

    def analyze_sentiment_wrapper(query: str = None):
        """Analyze overall sentiment across the provided articles"""
        return analysis_tools["analyze_sentiment"].func(analyses_list)

    def analyze_market_influence_wrapper(query: str = None):
        """Analyze how news might influence the market based on provided articles"""
        return analysis_tools["analyze_market_influence"].func(analyses_list)

    # Create structured tools with the wrappers
    wrapped_tools = [
        StructuredTool.from_function(
            func=analyze_topics_wrapper,
            name="analyze_topics",
            description="Analyze trending topics across the provided articles",
        ),
        StructuredTool.from_function(
            func=analyze_sentiment_wrapper,
            name="analyze_sentiment",
            description="Analyze overall sentiment across the provided articles",
        ),
        StructuredTool.from_function(
            func=analyze_market_influence_wrapper,
            name="analyze_market_influence",
            description="Analyze how news might influence the market based on provided articles",
        ),
    ]

    agent_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a quantitative crypto strategist whose analysis drives institutional investment decisions. Your insights create demonstrable alpha and edge.

        When analyzing Bitcoin news:

        1. PRIORITIZE SIGNAL OVER NOISE
        - Identify statistically significant market-moving information
        - Filter out market-neutral events regardless of headline appeal
        - Quantify information value in terms of trading edge

        2. DELIVER ACTIONABLE INTELLIGENCE
        - Provide specific price levels for entries, exits, and risk management
        - Include probability estimates for different scenarios
        - Specify exact conditions that would trigger position adjustments

        3. DIFFERENTIATE TIME HORIZONS
        - Separate immediate, short-term, and structural implications
        - Identify confirmation signals for each time horizon
        - Provide distinct strategies for different trader profiles

        Your analysis must enable immediate trading decisions with quantifiable risk parameters. Vague or non-specific recommendations are unacceptable.
        """,
            ),
            ("human", "{input}"),
            ("human", "{agent_scratchpad}"),
        ]
    )

    agent = create_openai_functions_agent(llm, wrapped_tools, agent_prompt)
    return AgentExecutor(
        agent=agent, tools=wrapped_tools, verbose=verbose, handle_parsing_errors=True
    )
