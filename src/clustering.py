"""
BERTopic clustering pipeline for fintech sentiment analysis.
Input: data/raw/combined_reviews.csv
Output: data/processed/reviews_with_topics.csv + topic_summary.csv
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from umap import UMAP

log = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# Generic sentiment words with no thematic specificity
_GENERIC_POSITIVE = frozenset({
    "great", "good", "love", "easy", "awesome", "wonderful", "best",
    "excellent", "amazing", "perfect", "fantastic", "convenient", "nice",
    "helpful", "fast", "quick",
})

# Words that anchor a topic to a real product theme (disqualify generic label)
_SPECIFIC_THEME = frozenset({
    "charge", "charged", "update", "card", "bank", "money", "pay", "credit",
    "score", "account", "fraud", "deposit", "fee", "fees", "interest",
    "refund", "cancel", "suspend", "close", "block", "login", "support",
    "cash", "chime", "affirm", "klarna", "chase",
})

# Themes to audit for BNPL-vs-cobrand behavioral signal
_BNPL_COBRAND_SIGNALS = {
    "switching_from_cards": {
        "label": "Switching away from credit cards / using BNPL at checkout",
        "keywords": [
            "switch", "switched", "instead", "replace", "replacing", "checkout",
            "no interest", "avoid", "credit card", "traditional",
        ],
    },
    "rewards_friction": {
        "label": "Cobrand rewards / points friction vs BNPL simplicity",
        "keywords": [
            "reward", "rewards", "points", "cashback", "cash back", "perks",
            "miles", "annual fee", "simple", "complicated", "friction",
        ],
    },
    "credit_trust": {
        "label": "Trust / credit-score impact: BNPL vs traditional credit",
        "keywords": [
            "credit score", "fico", "bureau", "hard pull", "soft pull",
            "hurt credit", "impact", "credit report", "report",
        ],
    },
    "primary_bank_replacement": {
        "label": "Deposit account (Chime / Cash App) replacing primary bank",
        "keywords": [
            "direct deposit", "paycheck", "primary", "main bank",
            "replacing", "switched to", "no fee", "free banking",
        ],
    },
}

# Reviews assigned this topic are generic-positive consolidations (not meaningful clusters)
GENERIC_TOPIC_LABEL = -2


# ── 1. Load & prep ───────────────────────────────────────────────────────────

def load_and_prep(path: Path | str | None = None) -> pd.DataFrame:
    if path is None:
        path = RAW_DIR / "combined_reviews.csv"

    df = pd.read_csv(path)
    raw_len = len(df)

    def _merge(row) -> str:
        title = str(row["title"] or "").strip()
        review = str(row["review"] or "").strip()
        title = "" if title.lower() in ("", "nan") else title
        review = "" if review.lower() in ("", "nan") else review
        return f"{title}. {review}".strip(". ") if title else review

    df["text"] = df.apply(_merge, axis=1)
    df = df[df["text"].str.len() > 10].copy().reset_index(drop=True)
    log.info(f"Prep: {raw_len} raw rows → {len(df)} usable docs")
    return df


# ── 2. Embed ─────────────────────────────────────────────────────────────────

def embed_docs(
    docs: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> np.ndarray:
    log.info(f"Embedding {len(docs)} docs with {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(docs, show_progress_bar=True, batch_size=batch_size)
    log.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings


# ── 3. Fit BERTopic ──────────────────────────────────────────────────────────

def fit_topics(
    docs: list[str],
    embeddings: np.ndarray,
    min_topic_size: int = 20,
    outlier_threshold: float = 0.20,
    target_topics: int | None = 10,
) -> tuple[BERTopic, list[int]]:
    """
    Fit BERTopic on the full corpus.  After initial clustering:
      1. Reduce outliers if fraction exceeds outlier_threshold.
      2. If target_topics is set and we have more clusters than that, BERTopic
         semantically merges the most similar pairs until target_topics remain.
    """
    umap_model = UMAP(
        n_components=5,
        n_neighbors=15,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        top_n_words=10,
        verbose=True,
    )

    log.info("Fitting BERTopic...")
    topics, _ = topic_model.fit_transform(docs, embeddings)

    n_actual = len([t for t in set(topics) if t != -1])
    n_outliers = sum(t == -1 for t in topics)
    outlier_frac = n_outliers / len(topics)
    log.info(f"Initial fit: {n_actual} topics, {n_outliers} outliers ({outlier_frac:.1%})")

    if outlier_frac > outlier_threshold:
        log.info(f"Outlier fraction {outlier_frac:.1%} > {outlier_threshold:.0%} — reducing...")
        topics = topic_model.reduce_outliers(
            docs, topics, strategy="embeddings", embeddings=embeddings
        )
        topic_model.update_topics(docs, topics=topics)
        n_after = sum(t == -1 for t in topics)
        log.info(f"After outlier reduction: {n_after} outliers ({n_after / len(topics):.1%})")

    n_current = len([t for t in set(topics) if t != -1])
    if target_topics and n_current > target_topics:
        log.info(f"Reducing from {n_current} → {target_topics} topics (semantic merging)...")
        topic_model.reduce_topics(docs, nr_topics=target_topics)
        topics = list(topic_model.topics_)
        n_final = len([t for t in set(topics) if t != -1])
        log.info(f"After reduce_topics: {n_final} topics remain")

    return topic_model, list(topics)


# ── 4. Identify and consolidate generic-positive topics ───────────────────────

def consolidate_generic_topics(
    topic_model: BERTopic,
    docs: list[str],
    topics: list[int],
    df: pd.DataFrame,
    short_word_threshold: int = 6,
    min_short_fraction: float = 0.55,
) -> tuple[BERTopic, list[int], int]:
    """
    Identify topics that are predominantly short, high-rating reviews with no
    thematic content (generic-positive noise), merge them into one, then relabel
    those documents as GENERIC_TOPIC_LABEL (-2) so they're excluded from the
    final topic table but still present in the CSV.

    A topic is 'generic-positive' if it meets BOTH:
      - ≥ min_short_fraction of its docs are ≤ short_word_threshold words AND rating ≥ 4
      - Keyword profile: ≥ 4 generic-positive keywords AND 0 specific-theme keywords

    Returns (updated_model, updated_topics, n_excluded_docs).
    """
    word_counts = df["text"].str.split().str.len().values
    ratings = pd.to_numeric(df["rating"], errors="coerce").fillna(0).values
    topics_arr = np.array(topics)

    all_topic_ids = sorted(t for t in set(topics) if t != -1)
    generic_ids = []

    for tid in all_topic_ids:
        mask = topics_arr == tid
        if not mask.any():
            continue

        # Keyword criteria
        kw_data = topic_model.get_topic(tid)
        if not kw_data:
            continue
        kw_words = [w.lower() for w, _ in kw_data[:10]]
        generic_kw_hits = sum(1 for w in kw_words if w in _GENERIC_POSITIVE)
        specific_kw_hits = sum(1 for w in kw_words if w in _SPECIFIC_THEME)

        # Doc-composition criteria: what fraction of this topic's docs are short + high-rating?
        short_positive = ((word_counts[mask] <= short_word_threshold) & (ratings[mask] >= 4))
        short_frac = short_positive.mean()

        is_generic = (generic_kw_hits >= 4 and specific_kw_hits == 0) or (short_frac >= min_short_fraction)

        status = "GENERIC" if is_generic else "thematic"
        log.info(
            f"  Topic {tid:3d} [{', '.join(kw_words[:6])}] "
            f"generic_kw={generic_kw_hits} specific_kw={specific_kw_hits} "
            f"short_frac={short_frac:.0%} → {status}"
        )

        if is_generic:
            generic_ids.append(tid)

    if not generic_ids:
        log.info("No generic-positive topics found — no consolidation needed")
        return topic_model, topics, 0

    # Count docs before merging
    n_excluded = int(sum(t in generic_ids for t in topics))
    log.info(f"Consolidating {len(generic_ids)} generic-positive topics ({n_excluded} docs): {generic_ids}")

    if len(generic_ids) > 1:
        topic_model.merge_topics(docs, generic_ids)
        topics = list(topic_model.topics_)

    # Relabel the consolidated generic topic as GENERIC_TOPIC_LABEL
    merged_id = generic_ids[0]
    topics = [GENERIC_TOPIC_LABEL if t == merged_id else t for t in topics]
    log.info(f"Relabelled Topic {merged_id} → {GENERIC_TOPIC_LABEL} (generic-positive bucket)")

    n_remaining = len([t for t in set(topics) if t not in (-1, GENERIC_TOPIC_LABEL)])
    log.info(f"Meaningful topics after consolidation: {n_remaining}")

    return topic_model, topics, n_excluded


# ── 5. Summarise ─────────────────────────────────────────────────────────────

def build_topic_summary(
    df: pd.DataFrame,
    topics: list[int],
    topic_model: BERTopic,
) -> pd.DataFrame:
    df = df.copy()
    df["topic"] = topics

    topic_info = topic_model.get_topic_info()
    # Include only real (non-generic, non-outlier) topic IDs
    meaningful_ids = sorted(
        t for t in df["topic"].unique()
        if t not in (-1, GENERIC_TOPIC_LABEL)
    )

    rows = []
    for tid in meaningful_ids:
        subset = df[df["topic"] == tid]
        if subset.empty:
            continue

        keywords = topic_model.get_topic(tid)
        label = ", ".join(w for w, _ in keywords[:6]) if keywords else "(no label)"

        app_counts = subset["app"].value_counts()
        app_pct = (app_counts / len(subset) * 100).round(0).astype(int)
        app_breakdown = "  ".join(f"{a}:{p}%" for a, p in app_pct.items())

        avg_rating = subset["rating"].dropna().mean()
        # Pick representative docs directly from the corpus (longest informative texts).
        # Avoids relying on topic_model.get_representative_docs which breaks after
        # reduce_topics + update_topics transformations.
        rep_docs = (
            subset.assign(_len=subset["text"].str.len())
            .sort_values("_len", ascending=False)["text"]
            .head(3)
            .tolist()
        )

        rows.append({
            "topic":         tid,
            "keywords":      label,
            "docs":          len(subset),
            "avg_rating":    round(avg_rating, 1) if not np.isnan(avg_rating) else None,
            "app_breakdown": app_breakdown,
            "rep_docs":      rep_docs,
        })

    # Append outlier row if any exist
    outlier_subset = df[df["topic"] == -1]
    if not outlier_subset.empty:
        avg_r = outlier_subset["rating"].dropna().mean()
        app_counts = outlier_subset["app"].value_counts()
        app_pct = (app_counts / len(outlier_subset) * 100).round(0).astype(int)
        rows.append({
            "topic":         -1,
            "keywords":      "(outliers / unassigned)",
            "docs":          len(outlier_subset),
            "avg_rating":    round(avg_r, 1) if not np.isnan(avg_r) else None,
            "app_breakdown": "  ".join(f"{a}:{p}%" for a, p in app_pct.items()),
            "rep_docs":      [],
        })

    return pd.DataFrame(rows).reset_index(drop=True)


def print_topic_report(summary: pd.DataFrame, n_generic_excluded: int = 0) -> None:
    if n_generic_excluded:
        print(
            f"\n[Note: {n_generic_excluded} generic-positive reviews consolidated and "
            f"excluded from the table below (saved to CSV as topic={GENERIC_TOPIC_LABEL})]"
        )

    display = summary[["topic", "keywords", "docs", "avg_rating", "app_breakdown"]].copy()
    display["keywords"] = display["keywords"].str[:55]
    print("\n" + display.to_string(index=False))

    print("\n" + "=" * 80)
    print("REPRESENTATIVE EXAMPLES PER TOPIC")
    print("=" * 80)
    for _, row in summary.iterrows():
        if row["topic"] < 0:
            continue
        kw = row["keywords"][:50]
        print(f"\n── Topic {row['topic']}: {kw}")
        rep_docs = list(row["rep_docs"]) if row["rep_docs"] is not None else []
        for i, doc in enumerate(rep_docs[:3], 1):
            excerpt = str(doc).replace("\n", " ")[:220]
            print(f"  [{i}] {excerpt}")


# ── 6. BNPL-vs-cobrand theme signal audit ────────────────────────────────────

def flag_bnpl_cobrand_themes(summary: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("BNPL-VS-COBRAND THEME SIGNAL AUDIT")
    print("=" * 80)

    for _, theme in _BNPL_COBRAND_SIGNALS.items():
        signal_words = theme["keywords"]
        print(f"\n[{theme['label']}]")

        hits = []
        for _, row in summary.iterrows():
            tid = row["topic"]
            if tid < 0:
                continue
            rep_docs = list(row["rep_docs"]) if row["rep_docs"] is not None else []
            # Search keywords + all 3 representative doc texts
            combined = (
                row["keywords"].lower()
                + " "
                + " ".join(str(d) for d in rep_docs).lower()
            )
            matched = [w for w in signal_words if w in combined]
            if matched:
                hits.append((tid, row["keywords"][:50], row["docs"], row["avg_rating"], matched[:4]))

        if hits:
            for tid, kw, docs, rating, matched in hits:
                print(f"  → Topic {tid} ({docs} docs, {rating}★): \"{kw}\"")
                print(f"     Signal words: {matched}")
        else:
            print("  ✗ No distinct cluster — signal likely absorbed into noise or catch-all topic")


# ── 7. Pipeline entry point ───────────────────────────────────────────────────

def run_pipeline(
    data_path: Path | str | None = None,
    min_topic_size: int = 20,
    target_topics: int = 10,
) -> tuple[BERTopic, pd.DataFrame, pd.DataFrame, int]:
    """
    Strategy:
      - Embed and cluster the full corpus so UMAP/HDBSCAN see the true density.
      - Use min_topic_size=20 (up from 15) for sharper initial clusters.
      - Reduce semantically to target_topics via reduce_topics.
      - Post-hoc identify and consolidate generic-positive catch-all topics;
        relabel those docs as topic=-2 and exclude them from the final table.
        This achieves the same noise-reduction goal as pre-filtering without
        destroying the UMAP density structure.
    """
    df = load_and_prep(data_path)
    docs = df["text"].tolist()

    embeddings = embed_docs(docs)
    topic_model, topics = fit_topics(
        docs, embeddings,
        min_topic_size=min_topic_size,
        target_topics=target_topics,
    )

    topic_model, topics, n_generic = consolidate_generic_topics(
        topic_model, docs, topics, df
    )

    df = df.copy()
    df["topic"] = topics

    summary = build_topic_summary(df, topics, topic_model)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / "reviews_with_topics.csv", index=False)
    summary.drop(columns=["rep_docs"]).to_csv(PROCESSED_DIR / "topic_summary.csv", index=False)
    log.info(f"Saved outputs to {PROCESSED_DIR}")

    return topic_model, df, summary, n_generic


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    topic_model, df, summary, n_generic = run_pipeline()
    print_topic_report(summary, n_generic_excluded=n_generic)
    flag_bnpl_cobrand_themes(summary)
