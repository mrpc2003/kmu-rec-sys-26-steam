#!/usr/bin/env python3
"""Weekly arXiv RecSys SOTA monitor for KMURecSys26 Steam — structural-fit triage.

Fulfills the standing instruction to CONTINUOUSLY explore the latest paper methodology, in a
durable form: scans recent arXiv cs.IR recommendation papers, drops anything from the
already-exhausted families, and surfaces only candidates that structurally FIT this
competition. Prints a compact markdown digest to stdout (the cron run reasons over it).

Competition structure (the fit filter):
  - SMALL, relatively DENSE bipartite graph (6710 users x 2437 items, 165k interactions)
  - per-user MEDIAN-4 candidate set, 1:1 balanced played/not-played RERANKING (test is 50/50)
  - test pairs carry NO text, NO sequence, NO timestamp (anonymous userID/gameID only)
  - public LB tracks the UNIFORM negative distribution (popularity debiasing HURTS)

Already exhausted (do NOT re-surface as novel):
  BPR-LightGCN, ALS/WMF, EASE, ItemKNN, SGL, SimGCL, XSimGCL, LightGCL, DirectAU,
  TF-IDF text, MiniLM/AlphaRec LM-semantic, Turbo-CF/GF-CF graph-filtering, DNS/MixGCF
  hard-negative, diffusion RecSys, GBDT/logreg stacker, LLM listwise rerankers, seed ensembling.

No network dependency beyond the arXiv export API (stdlib urllib). No keys. Read-only.
"""
from __future__ import annotations

import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

NS = {"a": "http://www.w3.org/2005/Atom"}

# Proper arXiv boolean queries (operators are NOT url-encoded; only term values are).
# Anchored on cs.IR so the API actually field-filters instead of returning generic recents.
QUERIES = [
    'cat:cs.IR AND abs:recommendation',
    'cat:cs.IR AND abs:"collaborative filtering"',
    'cat:cs.IR AND abs:"implicit feedback"',
    'cat:cs.IR AND abs:reranking',
]

# A paper must look like a RECOMMENDATION paper to pass the domain gate. Without this, generic
# cs.LG/physics/RLHF papers leak in via weak hint matches.
DOMAIN_TERMS = [
    "recommend", "collaborative filter", "recommender", "user-item", "user–item",
    "top-k recommend", "top-n recommend", "item ranking", "matrix factorization",
    "implicit feedback",
]

# Lowercase substrings that mark an ALREADY-EXHAUSTED family -> drop.
EXHAUSTED = [
    "lightgcn", "graph contrastive", "simgcl", "xsimgcl", "lightgcl", " sgl",
    "directau", "tf-idf", "tfidf", "minilm", "alpharec", "graph filter", "turbo-cf",
    "gf-cf", "hard negative", "mixgcf", "diffusion", "gradient boost", "xgboost",
    "lightgbm", "listwise rerank", "llm rerank", "language model rerank",
    "sequential recommend", "session-based", "session based", "knowledge graph",
    "social recommend", "multimodal", "multi-modal", "federated", "fairness", "survey",
    "cross-domain", "cold-start sequential",
]

# Recsys-specific structural-fit hints (generic words like 'alignment'/'calibrat' removed).
FIT_HINTS = [
    "reranking", "re-ranking", "pairwise ranking", "decision boundary", "negative sampling",
    "closed-form", "training-free", "spectral", "low-rank", "popularity bias", "debias",
    "ranking loss", "margin", "dense interaction", "small dataset", "calibrated ranking",
    "balanced", "objective function", "embedding collapse", "uniformity alignment",
]


def fetch(q: str, n: int = 15, tries: int = 3):
    # arXiv phrase-query format: spaces -> '+', double-quotes -> '%22'; keep ':' and field ops.
    safe_q = q.replace('"', '%22').replace(" ", "+")
    url = ("https://export.arxiv.org/api/query?search_query=" + safe_q
           + f"&sortBy=submittedDate&sortOrder=descending&max_results={n}")
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "kmu-recsys-monitor/1.0"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return ET.parse(r).getroot()
        except Exception as ex:
            print(f"  [retry {t+1}] {q[:40]} ({ex})", file=sys.stderr)
            time.sleep(8 * (t + 1))
    return None


def cats_of(entry) -> list[str]:
    return [c.get("term", "") for c in entry.findall("a:category", NS)]


def score_paper(title: str, summ: str, cats: list[str]) -> tuple[int, list[str]]:
    text = (title + " " + summ).lower()
    # Domain gate: must be a recommendation paper (cs.IR alone isn't enough — it also hosts IR/search).
    if not any(d in text for d in DOMAIN_TERMS):
        return -1, []
    if any(e in text for e in EXHAUSTED):
        return -1, []
    hits = [h for h in FIT_HINTS if h in text]
    return len(hits), hits


def main() -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(days=10)
    seen, cands = set(), []
    for q in QUERIES:
        root = fetch(q)
        if root is None:
            continue
        for e in root.findall("a:entry", NS):
            idnode = e.find("a:id", NS)
            tnode = e.find("a:title", NS)
            snode = e.find("a:summary", NS)
            pnode = e.find("a:published", NS)
            if idnode is None or tnode is None or snode is None or pnode is None:
                continue
            aid = idnode.text.split("/abs/")[-1]
            key = aid.split("v")[0]
            if key in seen:
                continue
            seen.add(key)
            try:
                pubdt = datetime.strptime(pnode.text[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except Exception:
                continue
            if pubdt < cutoff:
                continue
            title = " ".join(tnode.text.split())
            summ = " ".join(snode.text.split())
            fit, hits = score_paper(title, summ, cats_of(e))
            if fit <= 0:
                continue
            cands.append((fit, pnode.text[:10], aid, title, hits, summ[:260]))
        time.sleep(5)

    cands.sort(reverse=True)
    if not cands:
        print("NO_NEW_FIT_CANDIDATES")  # cron reasons: nothing structurally new this week
        return
    print(f"# arXiv RecSys monitor — {len(cands)} structurally-fit candidate(s) (last 10d)\n")
    for fit, pub, aid, title, hits, summ in cands[:8]:
        print(f"## [{pub}] {aid}  (fit={fit})")
        print(f"**{title}**")
        print(f"- fit hints: {', '.join(hits)}")
        print(f"- abstract: {summ}...")
        print(f"- https://arxiv.org/abs/{aid}\n")


if __name__ == "__main__":
    main()
