import pandas as pd


def merge_news_price_data(
    analysis_df: pd.DataFrame, price_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge news and price datasets

    Args:
        analysis_df (DataFrame): News analysis dataset
        price_df (DataFrame): Price dataset

    Returns:
        DataFrame: Merged dataset
    """
    # Convert date columns to datetime
    analysis_df = analysis_df.copy()
    price_df = price_df.copy()

    analysis_df["date"] = pd.to_datetime(analysis_df["date"])
    price_df["timestamp"] = pd.to_datetime(price_df["timestamp"])

    # Create date keys for merging
    analysis_df["merge_date"] = analysis_df["date"].dt.floor("H")
    price_df["merge_date"] = price_df["timestamp"].dt.floor("H")

    # Merge datasets
    merged_df = pd.merge(analysis_df, price_df, on="merge_date")
    return merged_df


def calculate_price_changes(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate price changes over different time windows

    Args:
        price_df (DataFrame): Price dataset

    Returns:
        DataFrame: Price data with change metrics
    """
    df = price_df.copy()
    df = df.sort_values("timestamp")

    # Calculate price changes
    df["price_pct_change_1h"] = df["close"].pct_change(1) * 100
    df["price_pct_change_24h"] = df["close"].pct_change(24) * 100
    df["price_pct_change_7d"] = df["close"].pct_change(168) * 100

    # Calculate volatility
    df["volatility_24h"] = (
        df["high"].rolling(24).max() / df["low"].rolling(24).min() - 1
    )

    return df
