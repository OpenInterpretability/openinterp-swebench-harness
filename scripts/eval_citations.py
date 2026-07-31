#!/usr/bin/env python3
"""Citation-attribution eval: every \\bibitem carrying an arXiv ID is checked against the
arXiv API for (a) the ID resolving at all and (b) the cited surname matching a real author.

Why this exists: a 25-agent self-audit (2026-07-29) found 4 bibitems attributing papers to
ORGANISATIONS ("Goodfire", "Scale AI", "JetBrains Research") that are in fact authored by
named individuals — in two already-published papers. Mis-attribution is the same failure
class as the BlackboxNLP template desk-reject: mechanical, checkable before submission,
and fatal to credibility without the science ever being read.

Run:  python scripts/eval_citations.py            # whole paper/ tree
      python scripts/eval_citations.py <file.tex> # one file (pre-submission gate)
Exit 0 = clean. Exit 1 = at least one unresolvable ID or attribution mismatch.
No credentials required. Network required (arXiv API, rate-limited to ~1 req per 3s).
"""
import re, sys, time, urllib.request, urllib.error, xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARXIV_RE = re.compile(r'arXiv:\s*([0-9]{4}\.[0-9]{4,5})', re.I)
BIBITEM_RE = re.compile(r'\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\}|\Z)', re.S)
# tokens that are NOT surnames — org names we must flag, plus latex noise
ORG_WORDS = {"inc", "ai", "research", "labs", "lab", "team", "google", "deepmind", "openai",
             "anthropic", "goodfire", "scale", "jetbrains", "meta", "microsoft", "nvidia"}

CACHE = Path(__file__).resolve().parent / ".arxiv_cache.json"

def arxiv_batch(ids):
    """Per-ID over HTTPS with an on-disk cache.

    Empirically (2026-07-29): the comma-separated id_list form returns HTTP 429 even for 3
    IDs, while single-ID HTTPS requests succeed; and http:// costs a 301 redirect. So: one
    ID per request, https, polite delay, and cache the result — arXiv metadata for a fixed
    ID never changes, so a second run is instant and hits the network zero times.
    """
    import json
    cache = {}
    if CACHE.exists():
        try: cache = json.loads(CACHE.read_text())
        except Exception: cache = {}
    ns, out, fetched = {"a": "http://www.w3.org/2005/Atom"}, {}, 0
    for aid in sorted(ids):
        if aid in cache:
            c = cache[aid]
            out[aid] = (c, None) if c else (None, "unresolved"); continue
        rec, err = None, None
        for attempt in range(3):
            try:
                url = f"https://export.arxiv.org/api/query?id_list={aid}&max_results=1"
                with urllib.request.urlopen(url, timeout=60) as r:
                    x = ET.fromstring(r.read())
                e = x.find("a:entry", ns)
                if e is not None:
                    authors = [(a.findtext("a:name", "", ns) or "").strip() for a in e.findall("a:author", ns)]
                    if authors:
                        rec = {"title": " ".join((e.findtext("a:title", "", ns) or "").split()),
                               "authors": authors}
                break
            except Exception as ex:
                err = "NETWORK"; time.sleep(4 * (attempt + 1))
        fetched += 1
        if err == "NETWORK" and rec is None:
            out[aid] = (None, "NETWORK")            # unchecked, NOT a citation defect
        else:
            cache[aid] = rec                        # cache negatives too (rec=None => unresolved)
            out[aid] = (rec, None) if rec else (None, "unresolved")
        time.sleep(3.2)
    try: CACHE.write_text(json.dumps(cache))
    except Exception: pass
    if fetched: print(f"  (fetched {fetched} new ID(s); {len(ids)-fetched} from cache)")
    return out

def surnames(authors):
    out = set()
    for a in authors:
        parts = [p for p in re.split(r"[\s,]+", a) if p]
        if parts: out.add(parts[-1].lower().strip(".").replace("\\", ""))
    return out

def cited_names(body):
    """Pull the plain-text author segment: everything before the first quote/title marker."""
    t = re.sub(r"\\[a-zA-Z]+\s*", " ", body)           # strip latex commands
    t = t.replace("~", " ").replace("{", " ").replace("}", " ")
    t = t.split("``")[0].split("''")[0]
    t = re.split(r"\.\s+[A-Z][a-z]+ing|\barXiv\b", t)[0]
    # NB: {1,} not {2,} — two-letter surnames are real (Du, Wu, Ni, Xu, Li) and dropping
    # them produced four false MISMATCHes on the first full-tree run.
    toks = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'-]{1,}", t)
    return [w.lower() for w in toks][:20]

def main():
    targets = [Path(sys.argv[1]).resolve()] if len(sys.argv) > 1 else sorted((ROOT / "paper").rglob("*.tex"))
    entries = []
    for tex in targets:
        try: src = tex.read_text(errors="ignore")
        except Exception: continue
        for key, body in BIBITEM_RE.findall(src):
            m = ARXIV_RE.search(body)
            if m: entries.append((tex, key, body, m.group(1)))
    seen = arxiv_batch({a for _, _, _, a in entries})
    problems, unchecked, checked = [], [], len(entries)
    for tex, key, body, aid in entries:
            meta, err = seen.get(aid, (None, "unresolved"))
            try: rel = tex.relative_to(ROOT)
            except ValueError: rel = tex.name
            if meta is None:
                # a network failure is NOT a citation defect — never conflate the two
                (unchecked if err == "NETWORK" else problems).append(
                    ("NETWORK" if err == "NETWORK" else "UNRESOLVED", rel, key, aid, err or "?")); continue
            real = surnames(meta["authors"])
            cited = cited_names(body)
            if any(c in real for c in cited):
                continue                                        # a real surname is present -> OK
            orgs = [c for c in cited if c in ORG_WORDS]
            detail = (f"cited as '{' '.join(cited[:5])}' but arXiv authors are "
                      f"{', '.join(meta['authors'][:3])}{' et al.' if len(meta['authors'])>3 else ''}")
            problems.append(("ORG-ATTRIB" if orgs else "MISMATCH", rel, key, aid, detail))

    print(f"=== eval_citations: {checked} arXiv-bearing bibitems across {len(targets)} .tex "
          f"({len(seen)} unique IDs) ===")
    if unchecked:
        print(f"  ⚠️  {len(unchecked)} NOT CHECKED (network/rate-limit, not a defect): "
              f"{', '.join(sorted({u[3] for u in unchecked}))[:120]}")
    if not problems:
        print("  ✅ every checked arXiv ID resolves and every citation names a real author")
        return 1 if unchecked else 0          # unchecked => inconclusive, not a pass
    for kind, f, key, aid, d in problems:
        print(f"  [{kind}] {f}  {{{key}}}  arXiv:{aid}\n      {d}")
    print(f"\n  {len(problems)} problem(s). Fix before submission/mint.")
    return 1

if __name__ == "__main__":
    sys.exit(main())
