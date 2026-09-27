#!/usr/bin/env python3
"""Generate the weekly arXiv digest for RL-for-LLM papers.

Stdlib only. Fetches the most recent submissions per category (cs.CL, cs.LG,
cs.AI), then filters locally: a paper is kept when its title+abstract matches
both an RL-core pattern and an LLM-side pattern. Phrase queries against the
legacy Atom API are unreliable (sporadic 406s), so category listing + local
filtering is the robust approach.

Writes ``digest/<YYYY>-W<WW>.md`` and rebuilds ``digest/README.md``.
Runs weekly via ``.github/workflows/arxiv-digest.yml``; also safe to run
locally (read-only against arXiv, writes only inside this repo).
"""

from __future__ import annotations

import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

ATOM = "{http://www.w3.org/2005/Atom}"
ROOT = Path(__file__).resolve().parent.parent
DIGEST_DIR = ROOT / "digest"
WINDOW_DAYS = 7
PER_CATEGORY = 200
RETRIES = 3

CATEGORIES = ["cs.CL", "cs.LG", "cs.AI"]

# A paper qualifies when it matches (RL core) AND (LLM side). The category
# listing guarantees neither, so both filters are mandatory.
RL_CORE = re.compile(
    r"reinforcement learning|\brlhf\b|\bgrpo\b|\brlvr\b|\bdpo\b"
    r"|policy optimization|preference optimization|reward model"
    r"|verifiable reward|\br1\b|reward shaping|reward hacking",
    re.IGNORECASE,
)
LLM_SIDE = re.compile(
    r"\bllm\b|\blrms?\b|language model|\breasoning\b|\bagentic?\b"
    r"|alignment|\bagent|instruction|chat|chain-of-thought|test-time"
    r"|tool[- ]?use|tool calling|multi-turn|\bmath\b|code generation",
    re.IGNORECASE,
)
TAGS = {
    "rlhf": r"\brlhf\b|reinforcement learning from human feedback",
    "grpo": r"\bgrpo\b|group relative policy optimization",
    "rlvr": r"\brlvr\b|verifiable reward",
    "dpo": r"\bdpo\b|direct preference optimization",
    "reward": r"reward model|reward modeling|reward design|reward shaping|reward hacking",
    "reasoning": r"\breasoning\b|chain-of-thought|test-time",
    "agentic": r"\bagentic\b|tool[- ]use|tool calling|multi-agent",
    "alignment": r"\balignment\b|safety|constitution",
    "policy-optimization": r"policy optimization|\bppo\b|on-policy|off-policy",
    "multimodal": r"multimodal|vision-language|\bvlm\b",
}
_TAG_PATTERNS = {name: re.compile(pat, re.IGNORECASE) for name, pat in TAGS.items()}


def fetch_category(category: str) -> list[dict]:
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {
            "search_query": f"cat:{category}",
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": PER_CATEGORY,
        }
    )
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "awesome-rl-roadmap-digest/0.1 (github.com; contact via repo issues)"
        },
    )
    last_err: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                root = ET.fromstring(resp.read())
            break
        except Exception as err:  # noqa: BLE001 - network layer varies by host
            last_err = err
            if attempt == RETRIES:
                raise
            time.sleep(3 * attempt)
    else:  # pragma: no cover - loop always breaks or raises
        raise RuntimeError(f"unreachable; last error: {last_err}")

    entries = []
    for e in root.findall(f"{ATOM}entry"):
        published = datetime.strptime(
            e.findtext(f"{ATOM}published", "1970-01-01T00:00:00Z"), "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=timezone.utc)
        entries.append(
            {
                "title": " ".join((e.findtext(f"{ATOM}title", "") or "").split()),
                "url": (e.findtext(f"{ATOM}id", "") or "").strip(),
                "published": published,
                "authors": [a.findtext(f"{ATOM}name", "") for a in e.findall(f"{ATOM}author")],
                "abstract": " ".join((e.findtext(f"{ATOM}summary", "") or "").split()),
                "cats": [c.get("term", "") for c in e.findall(f"{ATOM}category")],
            }
        )
    return entries


def select(all_entries: list[dict], cutoff: datetime) -> list[dict]:
    seen: set[str] = set()
    kept = []
    for e in all_entries:
        if e["url"] in seen or e["published"] < cutoff:
            continue
        seen.add(e["url"])
        hay = f"{e['title']} {e['abstract']}"
        if not (RL_CORE.search(hay) and LLM_SIDE.search(hay)):
            continue
        e["tags"] = sorted(name for name, pat in _TAG_PATTERNS.items() if pat.search(hay))
        kept.append(e)
    kept.sort(key=lambda e: (e["published"], e["title"]), reverse=True)
    return kept


def md_link(text: str, url: str) -> str:
    return f"[{text.replace('|', '\\|')}]({url})"


def render(kept: list[dict], scanned: int, week: str, now: datetime) -> str:
    lines = [
        f"# arXiv Weekly Digest — {week}",
        "",
        f"_Auto-generated {now.strftime('%Y-%m-%d %H:%M UTC')}. "
        f"{len(kept)} papers matched RL-for-LLM filters "
        f"(scanned {scanned} recent cs.CL/cs.LG/cs.AI submissions)._",
        "",
        "| Date | Paper | Authors | Tags |",
        "|------|-------|---------|------|",
    ]
    for e in kept:
        authors = e["authors"][0] if e["authors"] else "?"
        if len(e["authors"]) > 1:
            authors += " et al."
        lines.append(
            f"| {e['published'].strftime('%m-%d')} "
            f"| {md_link(e['title'], e['url'])} "
            f"| {authors.replace('|', '\\|')} "
            f"| {', '.join(e['tags']) or '—'} |"
        )
    lines += [
        "",
        "---",
        "",
        "Selection: recent cs.CL/cs.LG/cs.AI submissions whose title+abstract "
        "match both an RL-core and an LLM-side keyword pattern "
        "([`scripts/arxiv_digest.py`](../scripts/arxiv_digest.py)). "
        "False positives happen — curation happens in the stage pages of the [roadmap](../README.md).",
        "",
    ]
    return "\n".join(lines)


def rebuild_index(weeks: list[str]) -> None:
    lines = [
        "# 📬 arXiv Weekly Digest",
        "",
        "Every Monday an automated job scans recent arXiv submissions for "
        "RL-for-LLM papers and files them here, keyword-tagged. "
        "Watch this repo to get the digest in your feed.",
        "",
    ]
    if weeks:
        lines += [f"- [{w}]({w}.md)" for w in weeks]
    else:
        lines.append("_No digest yet — the first one lands on the Monday after publish._")
    lines += ["", "Back to the [roadmap](../README.md).", ""]
    (DIGEST_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=WINDOW_DAYS)
    iso = now.isocalendar()
    week = f"{iso.year}-W{iso.week:02d}"

    all_entries: list[dict] = []
    for cat in CATEGORIES:
        rows = fetch_category(cat)
        print(f"{cat}: fetched {len(rows)} recent submissions")
        all_entries.extend(rows)

    kept = select(all_entries, cutoff)
    print(f"Scanned {len(all_entries)} entries, kept {len(kept)} within {WINDOW_DAYS} days.")

    DIGEST_DIR.mkdir(parents=True, exist_ok=True)
    (DIGEST_DIR / f"{week}.md").write_text(
        render(kept, len(all_entries), week, now), encoding="utf-8"
    )

    weeks = sorted(
        (p.stem for p in DIGEST_DIR.glob("*.md") if p.stem != "README"),
        reverse=True,
    )
    rebuild_index(weeks)
    print(f"Wrote digest/{week}.md; index lists {len(weeks)} week(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
