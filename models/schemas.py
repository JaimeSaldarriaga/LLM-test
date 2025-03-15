from typing import Any, Dict, List, Optional

from langchain.pydantic_v1 import BaseModel, Field


class NewsAnalysis(BaseModel):
    """Structured analysis of a single news article with actionable trading insights"""

    sentiment: str = Field(
        description="The overall sentiment of the article: positive, negative, or neutral"
    )
    sentiment_score: float = Field(
        description="Sentiment score from -1 (very negative) to 1 (very positive)"
    )
    key_topics: List[str] = Field(
        description="List of main topics discussed in the article"
    )
    bitcoin_impact_potential: str = Field(
        description="Potential impact on Bitcoin price: high, medium, low, or none"
    )
    expected_price_movement: Dict[str, float] = Field(
        description="Expected price movement range in percentage (min/max)"
    )
    impact_probability: float = Field(
        description="Probability (0-100%) that this news will impact the market"
    )
    time_horizon: str = Field(
        description="Time horizon for expected impact: immediate (hours), short (days), medium (weeks), long (months)"
    )
    key_entities: List[Dict[str, Any]] = Field(
        description="Key entities mentioned with their market influence capability (1-10)"
    )
    credibility_score: float = Field(
        description="Evidence-based reliability score of the article (1-10)"
    )
    rumors_speculation: bool = Field(
        description="Whether the article contains rumors or speculation"
    )
    tech_focused: bool = Field(
        description="Whether the article is focused on technological aspects"
    )
    regulatory_focused: bool = Field(
        description="Whether the article is focused on regulations"
    )
    investment_advice: bool = Field(
        description="Whether the article provides investment advice"
    )
    catalytic_potential: float = Field(
        description="Potential to trigger broader market movements (0-10)"
    )
    trading_signal: Dict[str, Any] = Field(
        description="Trading signal with action (buy/sell/hold) and confidence level (0-100%)"
    )
    price_triggers: List[Dict[str, Any]] = Field(
        description="Specific conditions that would activate a trading response"
    )


class TopicTrendAnalysis(BaseModel):
    """Analysis of trending topics in cryptocurrency news with market impact assessment"""

    trending_topics: List[Dict[str, Any]] = Field(
        description="List of trending topics with their frequency, importance, and market impact potential"
    )
    topic_momentum: List[Dict[str, Any]] = Field(
        description="Rate of change for each trend (acceleration/deceleration)"
    )
    emerging_trends: List[Dict[str, Any]] = Field(
        description="List of emerging trends with growth rate and potential impact"
    )
    fading_trends: List[Dict[str, Any]] = Field(
        description="List of trends that are losing relevance with decay rate"
    )
    historical_precedent: List[Dict[str, Any]] = Field(
        description="Performance data from similar past trends"
    )
    counter_indicators: List[Dict[str, Any]] = Field(
        description="Early warning signals that would invalidate trends"
    )
    market_positioning: Dict[str, Any] = Field(
        description="How different investor segments are aligned with trends"
    )
    trend_summary: str = Field(description="Summary of the overall trend landscape")


class SentimentAnalysis(BaseModel):
    """Detailed sentiment analysis with trading implications"""

    overall_sentiment: str = Field(
        description="General market sentiment: bullish, bearish, or neutral"
    )
    sentiment_score: float = Field(
        description="Calibrated market sentiment score (1-100)"
    )
    statistical_distribution: Dict[str, float] = Field(
        description="Statistical distribution of sentiment readings"
    )
    sentiment_drivers: List[Dict[str, Any]] = Field(
        description="Key factors driving sentiment with weighted importance"
    )
    sentiment_trend: str = Field(
        description="Whether sentiment is improving, worsening, or stable"
    )
    sentiment_price_divergence: Dict[str, Any] = Field(
        description="Identification of potential reversals based on sentiment-price divergence"
    )
    sentiment_extremes: Dict[str, Any] = Field(
        description="Statistical outliers suggesting contrarian opportunities"
    )
    sentiment_segmentation: Dict[str, Any] = Field(
        description="Institutional vs retail sentiment divides"
    )
    historical_context: Dict[str, Any] = Field(
        description="Current sentiment compared to similar market phases"
    )
    conviction_level: Dict[str, float] = Field(
        description="Statistical confidence in sentiment assessment (0-100%)"
    )
    sentiment_summary: str = Field(description="Summary of the sentiment analysis")


class MarketInfluenceAnalysis(BaseModel):
    """Strategic analysis of market influence factors with probability assessment"""

    high_impact_topics: List[Dict[str, Any]] = Field(
        description="Topics likely to have high impact on Bitcoin price with probability estimates"
    )
    probability_assessment: Dict[str, float] = Field(
        description="Statistical likelihood estimates for different market scenarios"
    )
    volatility_projection: Dict[str, Any] = Field(
        description="Expected price movement ranges with confidence intervals"
    )
    rumor_assessment: Dict[str, Any] = Field(
        description="Assessment of rumors with credibility and potential impact"
    )
    market_drivers: List[Dict[str, Any]] = Field(
        description="Key market drivers ranked by importance and reliability"
    )
    catalytic_timeline: List[Dict[str, Any]] = Field(
        description="Sequence and timing of expected market-moving events"
    )


class PriceNewsCorrelationAnalysis(BaseModel):
    """Quantitative analysis of price-news relationships with statistical metrics"""

    correlation_patterns: List[Dict[str, Any]] = Field(
        description="Key correlation patterns with statistical significance (R-values)"
    )
    signal_lag_patterns: Dict[str, Any] = Field(
        description="Measurable timeframes between news events and price reactions"
    )
    news_reference: Dict[str, Any] = Field(
        description="Reference to news and the price changes they caused with statistical confidence"
    )
    topic_impacts: Dict[str, Dict[str, float]] = Field(
        description="Topic-specific price impacts with magnitude and probability"
    )
    sentiment_price_correlation: Dict[str, float] = Field(
        description="News sentiment vs. price direction correlation with confidence intervals"
    )
    false_signal_framework: Dict[str, Any] = Field(
        description="Framework for distinguishing noise from actionable intelligence"
    )
    volatility_prediction: Dict[str, Any] = Field(
        description="Expected price volatility based on news characteristics"
    )
    market_inefficiency: List[Dict[str, Any]] = Field(
        description="Opportunities where news are consistently mispriced by the market"
    )
    implementation_criteria: List[Dict[str, Any]] = Field(
        description="Specific criteria for strategy execution based on news events"
    )
    predictive_factors: List[Dict[str, Any]] = Field(
        description="News factors most predictive of price movements"
    )
    summary: str = Field(description="Summary of news-price relationship analysis")


class MarketActionRecommendation(BaseModel):
    """Detailed trading recommendations with risk parameters and execution strategy"""

    position_directive: Dict[str, Any] = Field(
        description="Clear action (strong buy/buy/neutral/sell/strong sell) with confidence percentage"
    )
    key_signals: List[Dict[str, Any]] = Field(
        description="Key signals to monitor with specific thresholds and importance"
    )
    interpretation_guide: Dict[str, str] = Field(
        description="How to interpret different news types for trading decisions"
    )
    priority_news_categories: List[Dict[str, Any]] = Field(
        description="News categories to give more weight to with specific thresholds"
    )
    trading_sentiment: Dict[str, Any] = Field(
        description="Bitcoin trading recommendation with supporting rationale"
    )
