"""
Review + sentiment data collection for fintech sentiment analysis.

Sources:
  - Google Play Store  (google-play-scraper)  — structured reviews with star rating
  - Reddit via PRAW                           — organic community sentiment

Apple's public RSS review API was deprecated in 2025 and returns empty feeds.
Google Play has the same target apps with large review volumes.

Target apps: Chime, Cash App, Affirm, Klarna + Chase cobrand as baseline.
"""

import logging
import os
import time
import pandas as pd
from datetime import datetime
from pathlib import Path

from google_play_scraper import reviews as gp_reviews, Sort
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# Google Play package IDs — verified 2025-06
APP_REGISTRY: dict[str, dict] = {
    "chime":    {"display_name": "Chime",    "play_id": "com.onedebit.chime",   "subreddits": ["chime", "personalfinance"]},
    "cash_app": {"display_name": "Cash App", "play_id": "com.squareup.cash",    "subreddits": ["CashApp", "personalfinance"]},
    "affirm":   {"display_name": "Affirm",   "play_id": "com.affirm.central",   "subreddits": ["affirm", "personalfinance"]},
    "klarna":   {"display_name": "Klarna",   "play_id": "com.myklarnamobile",   "subreddits": ["klarna", "personalfinance"]},
    "chase":    {"display_name": "Chase",    "play_id": "com.chase.sig.android","subreddits": ["Chase", "personalfinance"]},
}


# ── Google Play ──────────────────────────────────────────────────────────────

def fetch_play_reviews(
    app_key: str,
    how_many: int = 500,
    sleep_seconds: float = 1.0,
    country: str = "us",
) -> pd.DataFrame:
    """
    Pull Google Play reviews and save to data/raw/<app_key>_play.csv.

    google-play-scraper paginates internally via a continuation token.
    Max practical limit per call is ~200 before the API starts returning duplicates;
    for larger pulls we loop with continuation tokens.
    """
    if app_key not in APP_REGISTRY:
        raise ValueError(f"Unknown app key '{app_key}'. Choose from: {list(APP_REGISTRY)}")

    meta = APP_REGISTRY[app_key]
    play_id = meta["play_id"]
    log.info(f"[Play] Fetching up to {how_many} reviews for '{app_key}' ({play_id})...")

    all_reviews: list[dict] = []
    continuation_token = None
    batch_size = min(200, how_many)

    while len(all_reviews) < how_many:
        batch, continuation_token = gp_reviews(
            play_id,
            lang="en",
            country=country,
            sort=Sort.NEWEST,
            count=batch_size,
            continuation_token=continuation_token,
        )
        if not batch:
            break
        all_reviews.extend(batch)
        log.info(f"  fetched {len(batch)} → total {len(all_reviews)}")
        if continuation_token is None:
            break
        time.sleep(sleep_seconds)

    if not all_reviews:
        log.warning(f"[Play] No reviews returned for '{app_key}'.")
        return pd.DataFrame()

    df = pd.DataFrame(all_reviews[:how_many])
    df = df.rename(columns={
        "userName": "userName",
        "score": "rating",
        "at": "date",
        "content": "review",
    })
    df["title"] = ""  # Google Play reviews don't have titles
    df["app"] = app_key
    df["source"] = "google_play"
    df["fetched_at"] = datetime.utcnow().isoformat()

    keep = ["date", "userName", "rating", "title", "review", "app", "source", "fetched_at"]
    df = df[[c for c in keep if c in df.columns]]

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DATA_DIR / f"{app_key}_play.csv"
    df.to_csv(out_path, index=False)
    log.info(f"[Play] Saved {len(df)} reviews → {out_path}")
    return df


# ── Reddit ───────────────────────────────────────────────────────────────────

def _init_reddit():
    """Return a praw.Reddit instance from env vars, or None if creds are missing."""
    try:
        import praw
        client_id = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        user_agent = os.getenv("REDDIT_USER_AGENT", "fintech-sentiment-engine/0.1")
        if not (client_id and client_secret):
            log.warning("[Reddit] REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET not set — skipping Reddit.")
            return None
        return praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
    except ImportError:
        log.warning("[Reddit] praw not installed.")
        return None


def fetch_reddit_posts(
    app_key: str,
    post_limit: int = 200,
    comment_limit: int = 5,
) -> pd.DataFrame:
    """
    Pull Reddit posts (+ top comments) mentioning an app from its dedicated subreddits.
    Saves to data/raw/<app_key>_reddit.csv.

    Each row is one post-or-comment with columns aligned to the Play review schema
    so both sources can be concatenated for clustering.

    Requires REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env.
    """
    if app_key not in APP_REGISTRY:
        raise ValueError(f"Unknown app key '{app_key}'. Choose from: {list(APP_REGISTRY)}")

    reddit = _init_reddit()
    if reddit is None:
        return pd.DataFrame()

    meta = APP_REGISTRY[app_key]
    rows: list[dict] = []

    for sub_name in meta["subreddits"]:
        try:
            subreddit = reddit.subreddit(sub_name)
            # Pull from new + hot to get diverse recency coverage
            for feed in ("new", "hot"):
                posts = getattr(subreddit, feed)(limit=post_limit // 2)
                for post in posts:
                    rows.append({
                        "date":     datetime.utcfromtimestamp(post.created_utc).isoformat(),
                        "userName": str(post.author) if post.author else "[deleted]",
                        "rating":   None,
                        "title":    post.title,
                        "review":   post.selftext or post.title,
                        "app":      app_key,
                        "source":   f"reddit_r/{sub_name}",
                        "fetched_at": datetime.utcnow().isoformat(),
                    })
                    # Top comments add more signal per post
                    post.comments.replace_more(limit=0)
                    for comment in list(post.comments)[:comment_limit]:
                        rows.append({
                            "date":     datetime.utcfromtimestamp(comment.created_utc).isoformat(),
                            "userName": str(comment.author) if comment.author else "[deleted]",
                            "rating":   None,
                            "title":    "",
                            "review":   comment.body,
                            "app":      app_key,
                            "source":   f"reddit_r/{sub_name}_comment",
                            "fetched_at": datetime.utcnow().isoformat(),
                        })
            log.info(f"[Reddit] r/{sub_name}: {len(rows)} rows so far")
        except Exception as e:
            log.warning(f"[Reddit] r/{sub_name} failed: {e}")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DATA_DIR / f"{app_key}_reddit.csv"
    df.to_csv(out_path, index=False)
    log.info(f"[Reddit] Saved {len(df)} rows → {out_path}")
    return df


# ── Combined ─────────────────────────────────────────────────────────────────

def fetch_all_sources(
    app_key: str,
    play_reviews: int = 500,
    reddit_posts: int = 200,
) -> pd.DataFrame:
    """Fetch Google Play + Reddit for one app; return a single concatenated DataFrame."""
    frames = []

    play_df = fetch_play_reviews(app_key, how_many=play_reviews)
    if not play_df.empty:
        frames.append(play_df)

    reddit_df = fetch_reddit_posts(app_key, post_limit=reddit_posts)
    if not reddit_df.empty:
        frames.append(reddit_df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    out_path = RAW_DATA_DIR / f"{app_key}_combined.csv"
    combined.to_csv(out_path, index=False)
    log.info(f"[Combined] {app_key}: {len(combined)} total rows → {out_path}")
    return combined


def fetch_all(play_reviews: int = 500, reddit_posts: int = 200) -> dict[str, pd.DataFrame]:
    """Fetch all sources for every app in APP_REGISTRY."""
    return {
        key: fetch_all_sources(key, play_reviews=play_reviews, reddit_posts=reddit_posts)
        for key in APP_REGISTRY
    }


if __name__ == "__main__":
    df = fetch_play_reviews("chime", how_many=50)
    print(df[["date", "userName", "rating", "review"]].head())
    print(f"\nShape: {df.shape}")
    print(f"Rating dist:\n{df['rating'].value_counts().sort_index()}")
