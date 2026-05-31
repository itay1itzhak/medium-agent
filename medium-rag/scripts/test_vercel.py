"""
Vercel endpoint test script for the Medium RAG API.

Tests:
  GET  /api/stats   – configuration check
  POST /api/prompt  – 4 canonical RAG questions

Usage:
  python scripts/test_vercel.py <VERCEL_URL>
  VERCEL_URL=https://... python scripts/test_vercel.py
  python scripts/test_vercel.py   # prompts interactively
"""

import sys
import os
import json
import textwrap
import time

# ── Optional dotenv ──────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv

    _env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv not installed; fall back to real env vars

import requests

# ── ANSI colours ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

# ── Test questions ────────────────────────────────────────────────────────────
QUESTIONS = [
    "Which article explains how the attention mechanism works in transformers?",
    "List up to 3 articles about productivity and time management",
    "Summarize the key ideas about building machine learning models in production",
    "Recommend an article about startup fundraising and explain why",
]

WRAP = 72  # character wrap width for response text


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def color_status(code: int) -> str:
    """Return the status code as a coloured string."""
    if 200 <= code < 300:
        return f"{GREEN}{code} OK{RESET}"
    return f"{RED}{code} ERROR{RESET}"


def word_count(text: str) -> int:
    return len(text.split())


def _hr(char: str = "─", width: int = 52) -> str:
    return char * width


def divider(title: str = "", char: str = "═", width: int = 52) -> str:
    if title:
        return f"{YELLOW}{char * 4} {title} {char * max(0, width - len(title) - 6)}{RESET}"
    return f"{YELLOW}{char * width}{RESET}"


def print_error_response(resp: requests.Response) -> None:
    print(f"  {RED}Status : {resp.status_code}{RESET}")
    print(f"  {BOLD}Headers:{RESET}")
    for k, v in resp.headers.items():
        print(f"    {k}: {v}")
    print(f"\n  {BOLD}Body:{RESET}")
    try:
        body = resp.json()
        print(json.dumps(body, indent=4))
    except Exception:
        print(resp.text)


# ─────────────────────────────────────────────────────────────────────────────
# Test: GET /api/stats
# ─────────────────────────────────────────────────────────────────────────────

def test_stats(base_url: str) -> dict:
    """
    Returns a summary dict:
      {"status": int, "time_ms": int, "ok": bool}
    """
    url = base_url.rstrip("/") + "/api/stats"
    print(f"\n{divider('GET /api/stats')}")
    print(f"  {CYAN}URL: {url}{RESET}")

    t0 = time.perf_counter()
    try:
        resp = requests.get(url, timeout=30)
    except requests.RequestException as exc:
        print(f"  {RED}Request failed: {exc}{RESET}")
        return {"status": 0, "time_ms": 0, "ok": False}

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    print(f"  Status : {color_status(resp.status_code)}")
    print(f"  Time   : {elapsed_ms} ms")

    if resp.status_code != 200:
        print_error_response(resp)
        return {"status": resp.status_code, "time_ms": elapsed_ms, "ok": False}

    try:
        data = resp.json()
    except Exception:
        print(f"  {RED}Response is not valid JSON{RESET}")
        return {"status": resp.status_code, "time_ms": elapsed_ms, "ok": False}

    # Pretty-print
    print(f"\n  {BOLD}Response:{RESET}")
    print(json.dumps(data, indent=4))

    # Validate required keys
    required = {"chunk_size", "overlap_ratio", "top_k"}
    missing = required - set(data.keys())
    if missing:
        print(f"\n  {RED}Validation FAILED — missing keys: {missing}{RESET}")
        return {"status": resp.status_code, "time_ms": elapsed_ms, "ok": False}

    print(f"\n  {GREEN}Validation OK — chunk_size={data['chunk_size']}, "
          f"overlap_ratio={data['overlap_ratio']}, top_k={data['top_k']}{RESET}")
    return {"status": resp.status_code, "time_ms": elapsed_ms, "ok": True}


# ─────────────────────────────────────────────────────────────────────────────
# Test: POST /api/prompt
# ─────────────────────────────────────────────────────────────────────────────

def test_prompt(base_url: str, question: str, q_index: int) -> dict:
    """
    Returns a summary dict:
      {"status": int, "time_ms": int, "chunks": int, "ok": bool}
    """
    url = base_url.rstrip("/") + "/api/prompt"

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"\n{divider()}")
    print(f"{YELLOW}{BOLD}QUESTION {q_index}: {question}{RESET}")
    print(f"{divider()}")
    print(f"  {CYAN}URL: {url}{RESET}")

    payload = {"question": question}
    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=120)
    except requests.RequestException as exc:
        print(f"  {RED}Request failed: {exc}{RESET}")
        return {"status": 0, "time_ms": 0, "chunks": 0, "ok": False}

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    print(f"  Status : {color_status(resp.status_code)}")
    print(f"  Time   : {elapsed_ms} ms")

    if resp.status_code != 200:
        print_error_response(resp)
        return {"status": resp.status_code, "time_ms": elapsed_ms, "chunks": 0, "ok": False}

    try:
        data = resp.json()
    except Exception:
        print(f"  {RED}Response is not valid JSON{RESET}")
        return {"status": resp.status_code, "time_ms": elapsed_ms, "chunks": 0, "ok": False}

    context_items = data.get("context", [])
    n_chunks = len(context_items)

    # ── Context chunks ────────────────────────────────────────────────────────
    print(f"\n  Context chunks retrieved: {BOLD}{n_chunks}{RESET}")
    top3 = context_items[:3]
    box_width = 50
    print(f"  {CYAN}┌─ Top {len(top3)} chunks {'─' * max(0, box_width - 12)}┐{RESET}")
    for i, item in enumerate(top3, 1):
        score  = item.get("score", "?")
        title  = item.get("title", "Unknown")[:50]
        chunk  = item.get("chunk", "")
        chunk_idx = item.get("article_id", "?")
        preview = chunk[:100].replace("\n", " ").strip()
        print(f"  {CYAN}  {i}. [score={score:.4f}] \"{title}\" (chunk {chunk_idx}){RESET}")
        print(f"       \"{preview}...\"")
    print(f"  {CYAN}{'─' * (box_width + 4)}{RESET}")

    # ── LLM Response ─────────────────────────────────────────────────────────
    response_text = data.get("response", "")
    print(f"\n  {BOLD}{'─' * 3} LLM Response {'─' * 35}{RESET}")
    for line in textwrap.wrap(response_text, width=WRAP):
        print(f"  {line}")

    # ── Augmented Prompt Preview ──────────────────────────────────────────────
    aug = data.get("Augmented_prompt", {})
    sys_text  = aug.get("System", "")
    user_text = aug.get("User", "")
    sys_words  = word_count(sys_text)
    user_words = word_count(user_text)

    sys_preview  = sys_text[:80].replace("\n", " ").strip()
    user_preview = user_text[:80].replace("\n", " ").strip()

    print(f"\n  {BOLD}{'─' * 3} Augmented Prompt Preview {'─' * 23}{RESET}")
    print(f"  System ({sys_words} words): \"{sys_preview}...\"")
    print(f"  User   ({user_words} words): \"{user_preview}...\"")

    return {"status": resp.status_code, "time_ms": elapsed_ms, "chunks": n_chunks, "ok": True}


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(stats_result: dict, prompt_results: list) -> None:
    col_test   = 20
    col_status =  8
    col_chunks =  8
    col_time   =  9
    total_w = col_test + col_status + col_chunks + col_time + 5  # 4 separators + 1 padding

    sep_line = f"  ├{'─' * (col_test + 2)}┼{'─' * (col_status + 2)}┼{'─' * (col_chunks + 2)}┼{'─' * (col_time + 2)}┤"

    def row(label, status_code, chunks_str, time_str, ok):
        status_cell = f"{status_code}".center(col_status)
        chunks_cell = chunks_str.center(col_chunks)
        time_cell   = time_str.center(col_time)
        label_cell  = label.ljust(col_test)
        status_color = GREEN if ok else RED
        return (
            f"  │ {label_cell} │ "
            f"{status_color}{status_cell}{RESET} │ "
            f"{chunks_cell} │ "
            f"{time_cell} │"
        )

    header = (
        f"  │ {'Test'.ljust(col_test)} │ "
        f"{'Status'.center(col_status)} │ "
        f"{'Chunks'.center(col_chunks)} │ "
        f"{'Time'.center(col_time)} │"
    )
    top_border = f"  ┌{'─' * (col_test + 2)}┬{'─' * (col_status + 2)}┬{'─' * (col_chunks + 2)}┬{'─' * (col_time + 2)}┐"
    bot_border = f"  └{'─' * (col_test + 2)}┴{'─' * (col_status + 2)}┴{'─' * (col_chunks + 2)}┴{'─' * (col_time + 2)}┘"

    print(f"\n{YELLOW}{BOLD}{'═' * 52}")
    print("  SUMMARY")
    print(f"{'═' * 52}{RESET}")
    print(top_border)
    print(header)
    print(sep_line)

    # Stats row
    s = stats_result
    print(row("GET /api/stats", s["status"], "n/a", f"{s['time_ms']}ms", s["ok"]))

    for i, r in enumerate(prompt_results, 1):
        print(sep_line)
        print(row(f"POST Q{i}", r["status"], str(r["chunks"]), f"{r['time_ms']}ms", r["ok"]))

    print(bot_border)

    # Overall verdict
    all_ok = stats_result["ok"] and all(r["ok"] for r in prompt_results)
    if all_ok:
        print(f"\n  {GREEN}{BOLD}All tests passed.{RESET}")
    else:
        failed = ([] if stats_result["ok"] else ["GET /api/stats"]) + \
                 [f"POST Q{i+1}" for i, r in enumerate(prompt_results) if not r["ok"]]
        print(f"\n  {RED}{BOLD}Some tests FAILED: {', '.join(failed)}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def get_vercel_url() -> str:
    # 1. CLI argument
    if len(sys.argv) > 1:
        return sys.argv[1].rstrip("/")

    # 2. Environment variable (may have been loaded from .env above)
    env_url = os.environ.get("VERCEL_URL", "").strip()
    if env_url:
        return env_url.rstrip("/")

    # 3. Interactive prompt
    try:
        url = input("Enter your Vercel deployment URL: ").strip()
    except (EOFError, KeyboardInterrupt):
        print(f"\n{RED}Aborted.{RESET}")
        sys.exit(1)

    if not url:
        print(f"{RED}No URL provided. Exiting.{RESET}")
        sys.exit(1)

    return url.rstrip("/")


def main() -> None:
    base_url = get_vercel_url()

    print(f"\n{YELLOW}{BOLD}{'═' * 52}")
    print(f"  Vercel RAG API Test")
    print(f"  Target: {base_url}")
    print(f"{'═' * 52}{RESET}")

    # ── GET /api/stats ────────────────────────────────────────────────────────
    stats_result = test_stats(base_url)

    # ── POST /api/prompt × 4 ─────────────────────────────────────────────────
    prompt_results = []
    for idx, question in enumerate(QUESTIONS, 1):
        result = test_prompt(base_url, question, idx)
        prompt_results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(stats_result, prompt_results)


if __name__ == "__main__":
    main()
