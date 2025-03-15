import pandas as pd


def load_news_data(file_path):
    """
    Load and preprocess the cryptocurrency news dataset

    Args:
        file_path (str): Path to the news CSV file

    Returns:
        DataFrame: Preprocessed news data
    """
    news_df = pd.read_csv(file_path)

    # Basic preprocessing
    news_df["published_date"] = pd.to_datetime(news_df["published_date"])
    news_df = news_df.sort_values("published_date")

    return news_df


def load_prices_data(file_path: str) -> pd.DataFrame:
    """
    Fetch Bitcoin price data from an API

    Args:
        file_path (str): Path of the data file

    Returns:
        DataFrame: Bitcoin price data
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error fetching price data: {e}")
        return pd.DataFrame()


def create_sample(df: pd.DataFrame, sample_size: int, random_state: int = 42):
    """
    Create a random sample of the dataset

    Args:
        df (DataFrame): Input dataframe
        sample_size (int): Size of the sample
        random_state (int): Random seed for reproducibility

    Returns:
        DataFrame: Sample dataframe
    """
    sample_size = min(sample_size, len(df))
    return df.sample(sample_size, random_state=random_state)


def get_articles_by_date(
    date_str: str, news_df: pd.DataFrame, max_articles: int = 5
) -> list:
    """
    Get articles from a specific date.

    Args:
        date_str (str): Date in format YYYY-MM-DD
        news_df (pd.DataFrame): News dataframe
        max_articles (int): Maximum number of articles to return

    Returns:
        List[Dict]: List of articles
    """
    try:
        date = pd.to_datetime(date_str).date()
        filtered_df = news_df[news_df["published_date"].dt.date == date]

        if len(filtered_df) == 0:
            return []

        sample = filtered_df.sample(min(max_articles, len(filtered_df)))

        articles = []
        for _, row in sample.iterrows():
            articles.append(
                {
                    "title": row["title"],
                    "summary": row["summary"]
                    if not pd.isna(row["summary"])
                    else (
                        row["excerpt"]
                        if not pd.isna(row["excerpt"])
                        else "No content available"
                    ),
                    "date": row["published_date"].strftime("%Y-%m-%d"),
                }
            )

        return articles
    except Exception as e:
        return [{"error": str(e)}]
