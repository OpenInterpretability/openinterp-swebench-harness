#!/usr/bin/env python3
"""
Test the Liu et al. taxonomy judge on 3 trajectories (1 success / 1 wandering / 1 locked).

Builds a compressed trajectory summary from the Phase 6 N=99 captures on Drive,
fills the user prompt template, calls Claude via OpenRouter, parses JSON output,
and prints results.

Usage:
    OPENROUTER_API_KEY=sk-or-... python3 paper3_liu_judge_test.py
"""

import json
import os
import sys
from pathlib import Path
import urllib.request
import urllib.error

DRIVE_PHASE6 = Path(
    "/Users/caiovicentino/Library/CloudStorage/"
    "GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs/swebench_v6_phase6"
)
TRACES_DIR = DRIVE_PHASE6 / "traces"
TOOL_ENTROPY_PATH = Path(
    "/Volumes/SSD Major/fish/openinterp-swebench-harness/"
    "scripts/inflection_turn_out/v5_tool_entropy.json"
)
PROMPT_PATH = Path(
    "/Volumes/SSD Major/fish/openinterp-swebench-harness/"
    "scripts/paper3_liu_judge_prompt.txt"
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
# Per memory `feedback_use_openrouter_llm_api`: prefer OpenRouter with namespace anthropic/<...>.
# Pick sonnet for cheaper smoke (test only); the full N=99 run can use opus.
MODEL = os.environ.get("LIU_JUDGE_MODEL", "anthropic/claude-sonnet-4.6")
# Smoke-test fallback if OpenRouter key is dead (e.g., rotated):
FALLBACK_OPENAI_MODEL = os.environ.get("LIU_FALLBACK_OPENAI_MODEL", "gpt-4o-mini")


def load_prompt():
    text = PROMPT_PATH.read_text()
    # Split into SYSTEM + USER TEMPLATE on the marker.
    sys_marker = "### SYSTEM PROMPT ###"
    user_marker = "### USER PROMPT TEMPLATE ###"
    sys_start = text.index(sys_marker) + len(sys_marker)
    user_start = text.index(user_marker)
    system_prompt = text[sys_start:user_start].strip()
    user_template = text[user_start + len(user_marker):].strip()
    return system_prompt, user_template


def load_tool_entropy_index():
    rows = json.loads(TOOL_ENTROPY_PATH.read_text())
    return {r["iid"]: r for r in rows}


def pick_one_per_class(idx):
    picked = {}
    for iid, row in idx.items():
        c = row["sub_class"]
        if c not in picked:
            picked[c] = iid
        if len(picked) >= 3:
            break
    return picked


def compress_trajectory(trace_path, te_row, patch_path):
    raw = json.loads(trace_path.read_text())
    turns = raw.get("turns", [])
    n_turns = len(turns)

    def tool_call_summary(turn):
        tcs = turn.get("tool_calls", []) or []
        if not tcs:
            return "(no tool call)"
        names = ",".join(tc.get("name", "?") for tc in tcs)
        # add first 60 chars of first arg value for grounding
        try:
            first_args = tcs[0].get("arguments", {}) or {}
            arg_preview = "; ".join(
                f"{k}={str(v)[:60].replace(chr(10), ' ')}" for k, v in list(first_args.items())[:2]
            )
        except Exception:
            arg_preview = ""
        return f"{names} | {arg_preview}"

    first5 = "\n".join(
        f"  T{t.get('turn_idx', i)}: {tool_call_summary(t)}"
        for i, t in enumerate(turns[:5])
    )
    last10 = "\n".join(
        f"  T{t.get('turn_idx', i)}: {tool_call_summary(t)}"
        for i, t in enumerate(turns[-10:])
    )
    last_thinking = (turns[-1].get("thinking", "") or "")[:600] if turns else ""

    # issue text — best guess from first turn's prompt or content
    issue_text = ""
    if turns:
        ic = turns[0].get("content", "") or ""
        rt = turns[0].get("raw_response", "") or ""
        # the prompt isn't directly stored; use thinking/raw_response as proxy diagnostic
        issue_text = (ic or rt)[:800]

    patch_text = ""
    if patch_path.exists():
        try:
            patch_text = patch_path.read_text()[:1200]
        except Exception:
            patch_text = "<patch unreadable>"
    else:
        patch_text = "<NO PATCH EMITTED>"

    repo = trace_path.stem.split("__")[0].replace("instance_", "")

    fields = dict(
        iid=raw.get("instance_id", trace_path.stem),
        repo=repo,
        sub_class=te_row["sub_class"],
        n_turns=n_turns,
        emit_finish=te_row.get("emit_finish"),
        patch_n_bytes=te_row.get("patch_n_bytes", 0),
        tool_entropy_last10=te_row.get("tool_entropy_last10", 0.0),
        tool_diversity_last10=te_row.get("tool_diversity_last10", 0.0),
        bigram_repeat_rate=te_row.get("bigram_repeat_rate", 0.0),
        issue_text=issue_text or "(unavailable — only model thinking saved)",
        first5_tool_calls=first5 or "(none)",
        last10_tool_calls=last10 or "(none)",
        last_thinking=last_thinking or "(no thinking)",
        final_patch=patch_text,
    )
    return fields


def _post_json(url, headers, payload):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_openrouter(system_prompt, user_prompt, model):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 800,
    }
    try:
        return _post_json(
            OPENROUTER_URL,
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://openinterp.org",
                "X-Title": "paper3-liu-judge-test",
            },
            payload,
        )
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"OpenRouter HTTP {e.code}: {e.read().decode('utf-8', 'ignore')[:500]}"
        )


def call_openai(system_prompt, user_prompt, model):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 800,
        "response_format": {"type": "json_object"},
    }
    try:
        return _post_json(
            OPENAI_URL,
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            payload,
        )
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"OpenAI HTTP {e.code}: {e.read().decode('utf-8', 'ignore')[:500]}"
        )


def call_llm(system_prompt, user_prompt):
    """Try OpenRouter first (memory preference), fall back to OpenAI for smoke testing."""
    try:
        data = call_openrouter(system_prompt, user_prompt, MODEL)
        return ("openrouter", MODEL, data)
    except Exception as e:
        print(f"    (openrouter failed: {str(e)[:120]} -- falling back to OpenAI {FALLBACK_OPENAI_MODEL})")
        data = call_openai(system_prompt, user_prompt, FALLBACK_OPENAI_MODEL)
        return ("openai", FALLBACK_OPENAI_MODEL, data)


def parse_judge_output(content):
    # strip markdown fences if present
    c = content.strip()
    if c.startswith("```"):
        c = c.strip("`")
        if c.lower().startswith("json"):
            c = c[4:]
        c = c.strip()
    # try to find the first { ... } block
    try:
        first = c.index("{")
        last = c.rindex("}") + 1
        return json.loads(c[first:last])
    except Exception as e:
        return {"_parse_error": str(e), "_raw": content[:500]}


def main():
    system_prompt, user_template = load_prompt()
    te_idx = load_tool_entropy_index()
    picks = pick_one_per_class(te_idx)
    print(f"Picked: {picks}\n")

    results = {}
    total_in = 0
    total_out = 0
    for cls, iid in picks.items():
        trace_path = TRACES_DIR / f"{iid}.json"
        patch_path = TRACES_DIR / f"{iid}.patch"
        if not trace_path.exists():
            print(f"  ! missing trace: {trace_path}")
            continue
        fields = compress_trajectory(trace_path, te_idx[iid], patch_path)
        user_prompt = user_template.format(**fields)
        # rough token count (4 chars/token heuristic)
        in_chars = len(system_prompt) + len(user_prompt)
        in_tok = in_chars // 4
        print(f"=== {cls.upper()} :: {iid[:80]} ===")
        print(f"    input ~{in_tok} tok ({in_chars} chars)")

        try:
            provider, model_used, resp = call_llm(system_prompt, user_prompt)
        except Exception as e:
            print(f"    !! call failed: {e}")
            continue
        print(f"    provider={provider} model={model_used}")
        content = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {})
        out_tok = usage.get("completion_tokens", 0)
        in_tok_actual = usage.get("prompt_tokens", in_tok)
        total_in += in_tok_actual
        total_out += out_tok
        parsed = parse_judge_output(content)
        print(f"    raw output: {content.strip()[:300]}")
        print(f"    parsed: {json.dumps(parsed, indent=2)}")
        print(f"    usage: in={in_tok_actual} out={out_tok}\n")
        results[cls] = {"iid": iid, "parsed": parsed, "usage": usage}

    print("=== SUMMARY ===")
    print(json.dumps({k: v["parsed"] for k, v in results.items()}, indent=2))
    print(f"\nTotals (test): in={total_in} tok, out={total_out} tok")
    # projection: divide by len(results), multiply by 99
    if results:
        avg_in = total_in / len(results)
        avg_out = total_out / len(results)
        print(f"Per-trajectory avg: in={avg_in:.0f} tok, out={avg_out:.0f} tok")
        print(f"Projected N=99 ({MODEL}):")
        # Opus 4.7 hypothetical pricing $15/MTok in, $75/MTok out
        # Sonnet 4.6 hypothetical $3/MTok in, $15/MTok out
        for nm, pin, pout in [
            ("anthropic/claude-opus-4.7", 15.0, 75.0),
            ("anthropic/claude-sonnet-4.6", 3.0, 15.0),
        ]:
            cost = (avg_in * 99 * pin + avg_out * 99 * pout) / 1_000_000
            print(f"  {nm}: ${cost:.3f}")


if __name__ == "__main__":
    main()
