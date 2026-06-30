"""
Phase 3 — Claude API brief generator.
Input:  data/processed/topic_summary.csv + reviews_with_topics.csv
Output: PM-ready strategic narrative brief (string)

Topic labels are manually assigned from inspection of the BERTopic output and
are specific to the current pipeline run. Re-label if clustering is re-run with
different parameters.
"""

from pathlib import Path

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MODEL = "claude-sonnet-4-6"

# Human-readable theme labels keyed by BERTopic topic ID.
# IDs match the current pipeline run (min_topic_size=20, target_topics=10,
# generic consolidation → 9 meaningful topics: {0,1,3,4,5,6,7,8,9}).
TOPIC_LABELS: dict[int, str] = {
    0: "Card payments, refunds & merchant friction",
    1: "App update & login failures",
    3: "Account closures, dispute denials & credit score damage",
    4: "App quality — mixed satisfaction",
    5: "Chime product experience (SpotMe, early direct deposit)",
    6: "BNPL pay-over-time value proposition",
    7: "Banking fundamentals (fees, deposits, customer service)",
    8: "Cash App — fraud, unauthorized charges & deactivation",
    9: "Affirm — loan terms & interest rate friction",
}

SYSTEM_PROMPT = """\
You are a product analyst writing an internal strategic memo for a financial services product team.

You have analyzed Google Play Store reviews from five fintech apps: Chime, Cash App, Affirm, \
Klarna, and Chase. The reviews were clustered into semantic topics using BERTopic, producing \
the structured data below.

Your task: synthesize these findings into a sharp, evidence-based brief on competitive threats \
to cobrand credit card economics — specifically whether BNPL and neobank apps show substitution \
signals in user sentiment data.

Rules:
- Write as a sharp PM/analyst, not a data scientist or AI. The audience is your product team.
- Reason ACROSS topics to build a single coherent argument. Do NOT summarize topics one by one.
- Be honest about what the data shows vs. what it doesn't. If users aren't explicitly saying \
  "I switched from my Chase card to Affirm," don't claim that. Distinguish observation from inference.
- Build toward a specific, actionable point of view. End with a clear recommendation or a \
  pointed open question the team should resolve — not a list of possible next steps.
- Cite topic IDs and stats as evidence: e.g. "Topic 3 (account closures, 1.8★, 36% Affirm)." \
  Every claim needs a traceable data anchor.
- Structure: thesis first → 2–3 supporting points → one closing recommendation or open question.
- Length: 400–600 words. Be tight. No filler sentences. No bullet-point summaries.\
"""

_USER_FRAMING = (
    "Generate a strategic brief on what fintech app sentiment data reveals about "
    "competitive threats to cobrand credit card economics, specifically around "
    "BNPL and neobank substitution risk."
)

_DATA_NOTES = """\
--- Data notes ---
Source: Google Play Store only (no App Store, no Reddit in this run).
Corpus: ~850 usable reviews across Chime, Cash App, Affirm, Klarna, Chase (~200/app), 2025-06.
94 generic-positive short reviews excluded from topic modeling as noise (topic = -2).
BNPL-vs-cobrand behavioral signals (card switching, rewards friction, credit score impact, \
primary bank replacement) were audited but did not form isolated clusters — they appear as \
scattered sentences within broader topics rather than a dense behavioral signal.\
"""


# ── Data loading ─────────────────────────────────────────────────────────────

def load_topic_data(
    summary_path: Path | None = None,
    reviews_path: Path | None = None,
    n_rep_docs: int = 3,
) -> tuple[pd.DataFrame, dict[int, list[str]]]:
    """
    Load topic summary CSV and pull the N longest texts per topic as
    representative examples.

    Longest texts are a reliable proxy for informativeness: short reviews
    (the generic-positive noise) were already excluded during clustering.
    What remains at the top of the length distribution is specific, detailed
    feedback with real thematic content.

    Returns:
        summary   — DataFrame with topic stats + 'theme' label column
        rep_docs  — {topic_id: [doc1, doc2, doc3]}
    """
    summary_path = summary_path or PROCESSED_DIR / "topic_summary.csv"
    reviews_path = reviews_path or PROCESSED_DIR / "reviews_with_topics.csv"

    summary = pd.read_csv(summary_path)
    reviews = pd.read_csv(reviews_path)

    summary = summary[~summary["topic"].isin([-1, -2])].copy()
    summary["theme"] = summary["topic"].map(TOPIC_LABELS)

    rep_docs: dict[int, list[str]] = {}
    for tid in summary["topic"].tolist():
        subset = reviews[reviews["topic"] == tid].dropna(subset=["text"])
        reps = (
            subset.assign(_len=subset["text"].str.len())
            .sort_values("_len", ascending=False)["text"]
            .head(n_rep_docs)
            .tolist()
        )
        rep_docs[int(tid)] = reps

    return summary, rep_docs


# ── Prompt assembly ───────────────────────────────────────────────────────────

def _format_topic_context(
    summary: pd.DataFrame,
    rep_docs: dict[int, list[str]],
) -> str:
    lines: list[str] = []
    for _, row in summary.iterrows():
        tid = int(row["topic"])
        theme = row.get("theme") or row["keywords"]
        lines.append(f"\n--- Topic {tid}: {theme} ---")
        lines.append(
            f"  Reviews: {int(row['docs'])}  |  "
            f"Avg rating: {row['avg_rating']}/5  |  "
            f"App distribution: {row['app_breakdown']}"
        )
        lines.append("  Representative reviews:")
        for i, doc in enumerate(rep_docs.get(tid, []), 1):
            lines.append(f"    [{i}] {str(doc).replace(chr(10), ' ')[:350]}")
    return "\n".join(lines)


# ── Brief generation ──────────────────────────────────────────────────────────

def generate_brief(
    summary: pd.DataFrame | None = None,
    rep_docs: dict[int, list[str]] | None = None,
    model: str = MODEL,
    stream_to_stdout: bool = False,
) -> str:
    """
    Call Claude to synthesize a PM-ready strategic brief.

    Args:
        summary          — topic summary DataFrame (loads from disk if None)
        rep_docs         — {topic_id: [doc, ...]} (loads from disk if None)
        model            — Claude model ID
        stream_to_stdout — if True, streams tokens to stdout as they arrive
                           (useful for notebook / interactive use)

    Returns the complete brief as a plain string.
    """
    if summary is None or rep_docs is None:
        summary, rep_docs = load_topic_data()

    user_message = "\n\n".join([
        _USER_FRAMING,
        _format_topic_context(summary, rep_docs),
        _DATA_NOTES,
    ])

    client = Anthropic()

    with client.messages.stream(
        model=model,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        if stream_to_stdout:
            for chunk in stream.text_stream:
                print(chunk, end="", flush=True)
            print()
        final = stream.get_final_message()

    return "\n".join(
        block.text for block in final.content if block.type == "text"
    ).strip()


if __name__ == "__main__":
    brief = generate_brief(stream_to_stdout=True)
