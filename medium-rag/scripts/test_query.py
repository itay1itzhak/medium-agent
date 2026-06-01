"""
Test the deployed RAG app with a single user-provided query.

Usage:
  python scripts/test_query.py <VERCEL_URL> "<your question>"
  python scripts/test_query.py <VERCEL_URL>          # prompts for question
  VERCEL_URL=https://... python scripts/test_query.py "<your question>"
  python scripts/test_query.py <VERCEL_URL> "<question>" --json   # also print raw JSON

Stages:
  1. App status  – GET /api/stats
  2. Request     – POST /api/prompt with your question
  3. Context     – retrieved chunks
  4. Prompt      – augmented prompt sent to the LLM
  5. Answer      – final LLM response
"""

import sys
import os
import json
import textwrap
import time

try:
    from dotenv import load_dotenv
    _env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    load_dotenv(_env)
except ImportError:
    pass

import requests

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

WRAP = 76


def header(title: str) -> None:
    bar = "─" * (len(title) + 4)
    print(f"\n{YELLOW}{BOLD}┌{bar}┐")
    print(f"│  {title}  │")
    print(f"└{bar}┘{RESET}")


def ok(msg: str) -> None:
    print(f"  {GREEN}✓ {msg}{RESET}")


def err(msg: str) -> None:
    print(f"  {RED}✗ {msg}{RESET}")


def info(label: str, value: str) -> None:
    print(f"  {BOLD}{label:<18}{RESET} {value}")


def wrap_print(text: str, indent: int = 2) -> None:
    pad = " " * indent
    for line in textwrap.wrap(text, width=WRAP):
        print(f"{pad}{line}")


# ── Stage 1: GET /api/stats ───────────────────────────────────────────────────

def check_stats(base_url: str) -> bool:
    header("Stage 1 · App Status  GET /api/stats")
    url = base_url + "/api/stats"
    info("URL", url)

    t0 = time.perf_counter()
    try:
        resp = requests.get(url, timeout=30)
    except requests.RequestException as exc:
        err(f"Connection failed: {exc}")
        return False
    ms = int((time.perf_counter() - t0) * 1000)

    info("HTTP status", f"{resp.status_code}  ({ms} ms)")

    if resp.status_code != 200:
        err(f"Expected 200, got {resp.status_code}")
        print(f"\n  {DIM}Body:{RESET}\n  {resp.text[:400]}")
        return False

    try:
        data = resp.json()
    except Exception:
        err("Response is not valid JSON")
        return False

    required = {"chunk_size", "overlap_ratio", "top_k"}
    missing  = required - set(data.keys())
    if missing:
        err(f"Missing required keys: {missing}")
        return False

    info("chunk_size",    str(data["chunk_size"]))
    info("overlap_ratio", str(data["overlap_ratio"]))
    info("top_k",         str(data["top_k"]))
    ok("App is up and config looks valid")
    return True


# ── Stage 2-5: POST /api/prompt ──────────────────────────────────────────────

def run_query(base_url: str, question: str) -> tuple:
    # ── Stage 2: Request ─────────────────────────────────────────────────────
    header("Stage 2 · Sending Request  POST /api/prompt")
    url = base_url + "/api/prompt"
    info("URL",      url)
    info("Question", question)

    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json={"question": question}, timeout=120)
    except requests.RequestException as exc:
        err(f"Connection failed: {exc}")
        return False, None
    ms = int((time.perf_counter() - t0) * 1000)

    info("HTTP status", f"{resp.status_code}  ({ms} ms)")

    if resp.status_code != 200:
        err(f"Request failed with status {resp.status_code}")
        print(f"\n  {BOLD}Response headers:{RESET}")
        for k, v in resp.headers.items():
            print(f"    {k}: {v}")
        print(f"\n  {BOLD}Body:{RESET}")
        try:
            print(json.dumps(resp.json(), indent=4))
        except Exception:
            print(f"  {resp.text[:600]}")

        # Hint for common Vercel error shape
        try:
            body = resp.json()
            e = body.get("error", {})
            if isinstance(e, dict) and "id" in e:
                print(f"\n  {YELLOW}Hint: This is a Vercel-level error (ID: {e['id']}).")
                print(f"  Open Vercel dashboard → project → Functions → prompt")
                print(f"  and search for request ID {e['id']!r} to see the traceback.{RESET}")
            elif "error" in body:
                print(f"\n  {YELLOW}Hint: Application error — {body['error']}{RESET}")
        except Exception:
            pass
        return False, None

    try:
        data = resp.json()
    except Exception:
        err("Response is not valid JSON")
        print(f"  Raw body: {resp.text[:400]}")
        return False, None

    ok(f"Response received in {ms} ms")

    # ── Stage 3: Context ──────────────────────────────────────────────────────
    header("Stage 3 · Retrieved Context")
    chunks = data.get("context", [])
    info("Chunks retrieved", str(len(chunks)))

    if not chunks:
        print(f"  {YELLOW}No chunks returned — retrieval may have failed.{RESET}")
    else:
        for i, item in enumerate(chunks, 1):
            score   = item.get("score", "?")
            title   = item.get("title", "Unknown")
            art_id  = item.get("article_id", "?")
            preview = item.get("chunk", "")[:120].replace("\n", " ").strip()
            score_s = f"{score:.4f}" if isinstance(score, float) else str(score)
            print(f"\n  {CYAN}{i}. [{score_s}] \"{title}\"  (id={art_id}){RESET}")
            print(f"     \"{preview}...\"")

    # ── Stage 4: Augmented Prompt ─────────────────────────────────────────────
    header("Stage 4 · Augmented Prompt")
    aug       = data.get("Augmented_prompt", {})
    sys_text  = aug.get("System", "")
    user_text = aug.get("User",   "")

    if not aug:
        print(f"  {YELLOW}Augmented_prompt not found in response.{RESET}")
    else:
        words_sys  = len(sys_text.split())
        words_user = len(user_text.split())
        info("System prompt", f"{words_sys} words")
        wrap_print(sys_text[:300] + ("..." if len(sys_text) > 300 else ""), indent=4)
        print()
        info("User prompt",   f"{words_user} words")
        wrap_print(user_text[:300] + ("..." if len(user_text) > 300 else ""), indent=4)

    # ── Stage 5: Final Answer ─────────────────────────────────────────────────
    header("Stage 5 · Final Answer")
    response_text = (data.get("response") or "").strip()

    if not response_text:
        err("response field is empty — the LLM returned no visible text.")
        print(f"  {YELLOW}Check Vercel function logs for [prompt] ERROR or finish_reason.{RESET}")
        return False, data

    ok(f"{len(response_text.split())} words returned")
    print()
    wrap_print(response_text)
    return True, data


# ── Entry point ───────────────────────────────────────────────────────────────

def get_args():
    args = sys.argv[1:]

    print_json = "--json" in args
    if print_json:
        args = [a for a in args if a != "--json"]

    # Pull VERCEL_URL from env if not given as first positional arg
    if args and (args[0].startswith("http://") or args[0].startswith("https://")):
        base_url = args[0].rstrip("/")
        question = " ".join(args[1:]).strip() if len(args) > 1 else ""
    else:
        base_url = os.environ.get("VERCEL_URL", "").strip().rstrip("/")
        question = " ".join(args).strip()

    if not base_url:
        try:
            base_url = input("Enter your Vercel deployment URL: ").strip().rstrip("/")
        except (EOFError, KeyboardInterrupt):
            print(f"\n{RED}Aborted.{RESET}")
            sys.exit(1)

    if not base_url:
        print(f"{RED}No URL provided. Exiting.{RESET}")
        sys.exit(1)

    if not question:
        try:
            question = input("Enter your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{RED}Aborted.{RESET}")
            sys.exit(1)

    if not question:
        print(f"{RED}No question provided. Exiting.{RESET}")
        sys.exit(1)

    return base_url, question, print_json


def main():
    base_url, question, print_json = get_args()

    print(f"\n{YELLOW}{BOLD}{'═' * 54}")
    print(f"  RAG Query Test")
    print(f"  Target : {base_url}")
    print(f"{'═' * 54}{RESET}")

    stats_ok = check_stats(base_url)
    query_ok, raw_data = run_query(base_url, question)

    if print_json and raw_data is not None:
        header("Raw JSON Response  (--json)")
        print(json.dumps(raw_data, indent=2))

    print(f"\n{YELLOW}{BOLD}{'═' * 54}")
    print("  Result")
    print(f"{'═' * 54}{RESET}")
    print(f"  App status : {'OK' if stats_ok else f'{RED}FAILED{RESET}'}")
    print(f"  Query      : {'OK' if query_ok  else f'{RED}FAILED{RESET}'}")

    if stats_ok and query_ok:
        print(f"\n  {GREEN}{BOLD}All good.{RESET}")
    else:
        print(f"\n  {RED}{BOLD}Something went wrong — see the stage output above.{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
