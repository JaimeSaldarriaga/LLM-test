import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_sentiment_distribution(df: pd.DataFrame):
    """Plot the distribution of sentiment scores"""
    plt.figure(figsize=(18, 5))
    sns.histplot(df["sentiment_score"], bins=20, kde=True)
    plt.title("Distribution of Sentiment Scores in Cryptocurrency News")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Number of Articles")
    plt.axvline(x=0, color="r", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_sentiment_over_time(df: pd.DataFrame):
    """Plot sentiment scores over time"""
    plt.figure(figsize=(18, 5))
    plt.scatter(df["date"], df["sentiment_score"], alpha=0.6)
    plt.title("News Sentiment Over Time")
    plt.xlabel("Date")
    plt.ylabel("Sentiment Score")
    plt.axhline(y=0, color="r", linestyle="--", alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_top_topics(df: pd.DataFrame, top_n: int = 15):
    """Plot the most frequent topics"""
    # Extract all topics
    all_topics = []
    for topics in df["key_topics"]:
        all_topics.extend([topic.lower().strip() for topic in topics])

    # Count topic frequencies
    topic_counts = pd.Series(all_topics).value_counts().reset_index()
    topic_counts.columns = ["Topic", "Count"]
    topic_counts = topic_counts.sort_values("Count", ascending=False).head(top_n)

    plt.figure(figsize=(18, 5))
    sns.barplot(x="Count", y="Topic", data=topic_counts)
    plt.title(f"Top {top_n} Topics in Cryptocurrency News")
    plt.xlabel("Frequency")
    plt.ylabel("Topic")
    plt.tight_layout()
    plt.show()
