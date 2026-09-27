#!/usr/bin/env python3
"""Daily arXiv digest for RL-for-LLM papers (files organized per ISO week).

Stdlib only. Two data sources with automatic fallback, because the legacy
Atom API intermittently returns 406 from some networks (e.g. GitHub Actions
runners) while the RSS feed is served by separate infrastructure:

1. Atom API  — export.arxiv.org/api/query, recent submissions per category.
2. RSS feeds — export.arxiv.org/rss/<cat>, the current day's announcements.

A paper is kept when its title+abstract matches both an RL-core pattern and
an LLM-side pattern. Runs are idempotent: existing rows in the current week's
file are parsed back, merged by normalized arXiv id, and rewritten — so a
daily cron accumulates the week incrementally and source switches lose
nothing.

Writes ``digest/<YYYY>-W<WW>.md`` and rebuilds ``digest/README.md``.
Runs via ``.github/workflows/arxiv-digest.yml``; safe to run locally.
"""

from __future__ import annotations

import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

ATOM = "{http://www.w3.org/2005/Atom}"
DC = "{http://purl.org/dc/elements/1.1/}"
ROOT = Path(__file__).resolve().parent.parent
DIGEST_DIR = ROOT / "digest"
WINDOW_DAYS = 8  # keep-mode cutoff for Atom-sourced dates; merge dedupes anyway
PER_CATEGORY = 200
RETRIES = 3
UA = "awesome-rl-roadmap-digest/0.1 (github.com; contact via repo issues)"
CATEGORIES = ["cs.CL", "cs.LG", "cs.AI"]

# A paper qualifies when it matches (RL core) AND (LLM side).
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

_ID_RE = re.compile(r"abs/(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE)
_TITLE_SUFFIX_RE = re.compile(r"\s*\((arXiv:)?\d{4}\.\d{4,5}(v\d+)?(\s*\[[^\]]*\])?\)\s*$")
_TAG_STRIP_RE = re.compile(r"<[^>]+>")
_ROW_RE = re.compile(r"^\| (\d{2}-\d{2}) \| \[(.+?)\]\((.+?)\) \| (.+?) \| (.+?) \|$")


def http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    last_err: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except Exception as err:  # noqa: BLE001 - network layer varies by host
            last_err = err
            if attempt == RETRIES:
                raise
            time.sleep(3 * attempt)
    raise RuntimeError(f"unreachable; last error: {last_err}")


def arxiv_id(url: str) -> str:
    m = _ID_RE.search(url)
    return m.group(1) if m else url


def tags_for(hay: str) -> list[str]:
    return sorted(name for name, pat in _TAG_PATTERNS.items() if pat.search(hay))


def qualifies(hay: str) -> bool:
    return bool(RL_CORE.search(hay) and LLM_SIDE.search(hay))


def first_author(authors: list[str] | str) -> str:
    if isinstance(authors, str):
        authors = [a.strip() for a in authors.split(",") if a.strip()]
    if not authors:
        return "?"
    return authors[0] + (" et al." if len(authors) > 1 else "")


# ---------------------------------------------------------------- source 1


def fetch_atom() -> list[dict]:
    entries: list[dict] = []
    for cat in CATEGORIES:
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {
                "search_query": f"cat:{cat}",
                "sortBy": "submittedDate",
                "sortOrder": "descending",
                "max_results": PER_CATEGORY,
            }
        )
        root = ET.fromstring(http_get(url))
        for e in root.findall(f"{ATOM}entry"):
            raw_id = e.findtext(f"{ATOM}id", "") or ""
            published = datetime.strptime(
                e.findtext(f"{ATOM}published", "1970-01-01T00:00:00Z"), "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=timezone.utc)
            entries.append(
                {
                    "id": arxiv_id(raw_id),
                    "title": _TITLE_SUFFIX_RE.sub(
                        "", " ".join((e.findtext(f"{ATOM}title", "") or "").split())
                    ),
                    "url": f"https://arxiv.org/abs/{arxiv_id(raw_id)}",
                    "date": published.strftime("%m-%d"),
                    "ymd": published.date().isoformat(),
                    "authors": [a.findtext(f"{ATOM}name", "") for a in e.findall(f"{ATOM}author")],
                    "abstract": " ".join((e.findtext(f"{ATOM}summary", "") or "").split()),
                }
            )
    return entries


# ---------------------------------------------------------------- source 2


def fetch_rss() -> list[dict]:
    entries: list[dict] = []
    for cat in CATEGORIES:
        root = ET.fromstring(http_get(f"https://export.arxiv.org/rss/{cat}"))
        pub = root.findtext("channel/pubDate", "") or ""
        try:
            pub_dt = parsedate_to_datetime(pub)
        except (TypeError, ValueError):
            pub_dt = datetime.now(timezone.utc)
        for item in root.findall("channel/item"):
            link = (item.findtext("link", "") or "").strip()
            title = _TITLE_SUFFIX_RE.sub("", " ".join((item.findtext("title", "") or "").split()))
            desc = _TAG_STRIP_RE.sub(" ", item.findtext("description", "") or "")
            entries.append(
                {
                    "id": arxiv_id(link),
                    "title": title,
                    "url": f"https://arxiv.org/abs/{arxiv_id(link)}",
                    "date": pub_dt.strftime("%m-%d"),
                    "ymd": pub_dt.date().isoformat(),
                    "authors": item.findtext(f"{DC}creator", "") or "",
                    "abstract": " ".join(desc.split()),
                }
            )
    return entries


# ---------------------------------------------------------------- pipeline


def parse_existing(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _ROW_RE.match(line.strip())
        if not m:
            continue
        date, title, url, authors, tags = m.groups()
        tags = [] if tags == "—" else [t.strip() for t in tags.split(",") if t.strip()]
        rows[arxiv_id(url)] = {
            "id": arxiv_id(url),
            "title": title,
            "url": f"https://arxiv.org/abs/{arxiv_id(url)}",
            "date": date,
            "authors": authors,
            "tags": tags,
        }
    return rows


def collect_fresh(cutoff: datetime) -> dict[str, dict]:
    fresh: dict[str, dict] = {}
    source = "atom"
    try:
        raw = fetch_atom()
    except Exception as err:  # noqa: BLE001 - fall back to the other source
        print(f"Atom API failed ({err!r}); falling back to RSS feeds.")
        source = "rss"
        raw = fetch_rss()

    kept = 0
    for e in raw:
        hay = f"{e['title']} {e['abstract']}"
        if e["id"] not in fresh and qualifies(hay):
            fresh[e["id"]] = {
                **e,
                "authors": first_author(e["authors"]),
                "tags": tags_for(hay),
            }
            kept += 1
    print(f"Source={source}: {len(raw)} fetched, {kept} matched RL-for-LLM filters.")
    return fresh


def render(rows: dict[str, dict], week: str, now: datetime, sources_note: str) -> str:
    ordered = sorted(rows.values(), key=lambda r: (r["date"], r["title"]), reverse=True)
    lines = [
        f"# arXiv Digest — {week} (daily, cumulative)",
        "",
        f"_Auto-generated; last update {now.strftime('%Y-%m-%d %H:%M UTC')}. "
        f"{len(ordered)} papers this week. {sources_note}_",
        "",
        "| Date | Paper | Authors | Tags |",
        "|------|-------|---------|------|",
    ]
    for r in ordered:
        title = r["title"].replace("|", "\\|")
        authors = r["authors"].replace("|", "\\|")
        lines.append(
            f"| {r['date']} | [{title}]({r['url']}) | {authors} "
            f"| {', '.join(r['tags']) or '—'} |"
        )
    lines += [
        "",
        "---",
        "",
        "Selection: recent cs.CL/cs.LG/cs.AI arXiv submissions whose title+abstract "
        "match both an RL-core and an LLM-side keyword pattern "
        "([`scripts/arxiv_digest.py`](../scripts/arxiv_digest.py)). "
        "False positives happen — curation happens in the stage pages of the [roadmap](../README.md).",
        "",
    ]
    return "\n".join(lines)


def rebuild_index(weeks: list[str]) -> None:
    lines = [
        "# 📬 arXiv Daily Digest",
        "",
        "Every weekday an automated job scans new arXiv submissions for RL-for-LLM "
        "papers and files them here, keyword-tagged, one cumulative file per week. "
        "Watch this repo to get fresh papers in your feed.",
        "",
    ]
    if weeks:
        lines += [f"- [{w}]({w}.md)" for w in weeks]
    else:
        lines.append("_No digest yet — the first one lands after the next weekday run._")
    lines += ["", "Back to the [roadmap](../README.md).", ""]
    (DIGEST_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=WINDOW_DAYS)
    iso = now.isocalendar()
    week = f"{iso.year}-W{iso.week:02d}"

    DIGEST_DIR.mkdir(parents=True, exist_ok=True)
    week_path = DIGEST_DIR / f"{week}.md"
    rows = parse_existing(week_path)
    before = len(rows)

    fresh = collect_fresh(cutoff)
    cutoff_ymd = cutoff.date().isoformat()
    rows.update({k: v for k, v in fresh.items() if v["ymd"] >= cutoff_ymd})

    note = f"(Merged {len(rows) - before} new of {len(fresh)} matched.)"
    week_path.write_text(render(rows, week, now, note), encoding="utf-8")

    weeks = sorted(
        (p.stem for p in DIGEST_DIR.glob("*.md") if p.stem != "README"), reverse=True
    )
    rebuild_index(weeks)
    print(f"Wrote digest/{week}.md: {before} existing + {len(rows) - before} new = {len(rows)} rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
