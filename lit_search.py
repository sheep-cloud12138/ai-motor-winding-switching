"""Small literature search helper: arXiv / Crossref / OpenAlex.

Usage:
    python lit_search.py "winding reconfiguration" --source arxiv --n 20
"""

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def clean(text):
    return re.sub(r"\s+", " ", (text or "")).strip()


def arxiv(query, n=20):
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": query, "start": 0, "max_results": n}
    )
    root = ET.fromstring(requests.get(url, timeout=60).content)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in root.findall("a:entry", ns):
        out.append(
            {
                "id": clean(e.findtext("a:id", "", ns)),
                "title": clean(e.findtext("a:title", "", ns)),
                "published": clean(e.findtext("a:published", "", ns))[:10],
                "abstract": clean(e.findtext("a:summary", "", ns)),
                "authors": [
                    clean(a.findtext("a:name", "", ns)) for a in e.findall("a:author", ns)
                ],
                "pdf": next(
                    (
                        l.get("href")
                        for l in e.findall("a:link", ns)
                        if l.get("title") == "pdf"
                    ),
                    "",
                ),
            }
        )
    return out


def crossref(query, n=20):
    params = {
        "query.bibliographic": query,
        "rows": n,
        "select": "DOI,title,container-title,issued,author,abstract,is-referenced-by-count,URL",
    }
    for attempt in range(5):
        r = requests.get("https://api.crossref.org/works", params=params, timeout=60)
        if r.status_code != 429:
            r.raise_for_status()
            break
        time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError("Crossref rate limited after retries")
    out = []
    for i in r.json()["message"]["items"]:
        out.append(
            {
                "doi": i.get("DOI", ""),
                "title": clean((i.get("title") or [""])[0]),
                "venue": clean((i.get("container-title") or [""])[0]),
                "year": (i.get("issued", {}).get("date-parts") or [[""]])[0][0],
                "cites": i.get("is-referenced-by-count", 0),
                "abstract": clean(i.get("abstract", "")),
                "url": i.get("URL", ""),
            }
        )
    return out


def s2(query, n=20):
    r = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={
            "query": query,
            "limit": n,
            "fields": "title,abstract,year,venue,externalIds,openAccessPdf,citationCount,authors",
        },
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError(f"S2 HTTP {r.status_code}: {r.text[:200]}")
    out = []
    for i in r.json().get("data", []):
        out.append(
            {
                "doi": (i.get("externalIds") or {}).get("DOI", ""),
                "arxiv": (i.get("externalIds") or {}).get("ArXiv", ""),
                "title": clean(i.get("title", "")),
                "venue": clean(i.get("venue", "")),
                "year": i.get("year", ""),
                "cites": i.get("citationCount", 0),
                "abstract": clean(i.get("abstract", "")),
                "oa_url": (i.get("openAccessPdf") or {}).get("url", ""),
                "authors": [a.get("name", "") for a in (i.get("authors") or [])],
            }
        )
    return out


def openalex(query, n=20):
    r = requests.get(
        "https://api.openalex.org/works",
        params={"search": query, "per-page": n},
        timeout=60,
    )
    r.raise_for_status()
    out = []
    for i in r.json().get("results", []):
        inv = i.get("abstract_inverted_index") or {}
        words = []
        for w, pos in inv.items():
            for p in pos:
                words.append((p, w))
        abstract = " ".join(w for _, w in sorted(words))
        out.append(
            {
                "doi": (i.get("doi") or "").replace("https://doi.org/", ""),
                "title": clean(i.get("title", "")),
                "year": i.get("publication_year", ""),
                "cites": i.get("cited_by_count", 0),
                "venue": clean((i.get("primary_location") or {}).get("source", {}).get("display_name", "") if (i.get("primary_location") or {}).get("source") else ""),
                "abstract": clean(abstract),
                "oa_url": ((i.get("best_oa_location") or {}).get("pdf_url") or ""),
                "url": i.get("id", ""),
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--source", default="arxiv", choices=["arxiv", "crossref", "openalex", "s2"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    fn = {"arxiv": arxiv, "crossref": crossref, "openalex": openalex, "s2": s2}[args.source]
    rows = fn(args.query, args.n)
    for i, r in enumerate(rows, 1):
        title = r.get("title", "")
        year = r.get("year") or r.get("published", "")
        venue = r.get("venue", "")
        print(f"[{i:02d}] {year} | {title}")
        if venue:
            print(f"     {venue}")
        key = r.get("id") or r.get("doi") or r.get("url")
        if key:
            print(f"     {key}")
        if r.get("abstract"):
            print(f"     {r['abstract'][:500]}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
