import json

import pandas as pd
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool

from tools.relationship_analysis_tools import create_relationship_analysis_tools


def create_relationship_analysis_agent(llm, merged_df, verbose):
    """
    Create an agent for analyzing relationships between news and prices

    Args:
        llm: The language model to use
        merged_df: Merged news and price dataframe
        news_tools: News analysis tools
        relationship_tools: Relationship analysis tools

    Returns:
        AgentExecutor: Agent for relationship analysis
    """

    relationship_analysis_tools = create_relationship_analysis_tools(llm)

    # If article_analyses is a DataFrame, convert it properly
    if isinstance(merged_df, pd.DataFrame):
        # Convert to records and handle date/timestamp conversion
        merged_price_news_data = json.loads(
            merged_df.to_json(orient="records", date_format="iso")
        )
    else:
        # If it's already a dict or list, just ensure it's JSON serializable
        try:
            # Test if it can be serialized
            json.dumps(merged_df)
            merged_price_news_data = merged_df
        except TypeError:
            # If not, convert any problematic elements
            if isinstance(merged_df, dict):
                merged_price_news_data = json.loads(
                    pd.json_normalize(merged_df).to_json(orient="records")
                )[0]
            else:
                merged_price_news_data = json.loads(
                    pd.DataFrame(merged_df).to_json(orient="records")
                )

    # Create wrapped tools that include the article analyses
    def analyze_price_news_correlation_wrapper(query: str = None):
        """Analyze correlation between news and prices"""
        return relationship_analysis_tools["analyze_price_news_correlation"].func(
            merged_price_news_data
        )

    def generate_trading_insights_wrapper(query: str = None):
        """Generate trading insights based on the analyzed correlation"""
        return relationship_analysis_tools["generate_trading_insights"].func(
            merged_price_news_data
        )

    # Create structured tools with the wrappers
    wrapped_tools = [
        StructuredTool.from_function(
            func=analyze_price_news_correlation_wrapper,
            name="analyze_price_news_correlation",
            description="Analyze correlation between news and prices",
        ),
        StructuredTool.from_function(
            func=generate_trading_insights_wrapper,
            name="generate_trading_insights",
            description="Generate trading insights based on the analyzed correlation",
        ),
    ]

    agent_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a quantitative investment analyst specializing in extracting tradable edge from crypto market inefficiencies.

When analyzing news-price relationships:

1. IDENTIFY EXPLOITABLE PATTERNS
   - Calculate statistical edge (win rate, expected value)
   - Measure persistence and decay of different signal types
   - Quantify market overreaction and underreaction scenarios

2. DEVELOP IMPLEMENTATION FRAMEWORK
   - Specify exact entry/exit execution methodology
   - Define position sizing model with risk parameters
   - Provide scenario-based adjustment triggers

3. DELIVER PORTFOLIO-LEVEL INSIGHTS
   - Translate findings into allocation recommendations
   - Identify correlation with existing strategies and assets
   - Provide portfolio-level risk metrics for implementation

Your recommendations must be specific enough to be programmatically implemented and backtested. Include all parameters required for strategy execution.
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
