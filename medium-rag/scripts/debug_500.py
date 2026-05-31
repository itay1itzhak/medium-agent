"""
Quick diagnostic script for the Vercel 500 error.

Usage:
  python scripts/debug_500.py https://medium-agent-93s6.vercel.app
  VERCEL_URL=https://medium-agent-93s6.vercel.app python scripts/debug_500.py

What it does:
  1. Hits POST /api/prompt with a minimal question and dumps the full response.
  2. Reports whether the error is Vercel-level (500 format with "id") or
     application-level (503 with "error" string).
  3. Prints exact headers, status, and body so the step=N log from
     api/prompt.py can be cross-checked with the Vercel function logs.

After running:
  - Check https://vercel.com/dashboard → your project → Functions → prompt logs
    to see which step=N OK / FAIL line is the last one printed.
"""

import sys
import os
import json
import time

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass

import requests

RED   = "\033[91m"
GREEN = "\033[92m"
CYAN  = "\033[96m"
BOLD  = "\033[1m"
RESET = "\033[0m"

QUESTION = "What is the attention mechanism in transformers?"


def main():
    if len(sys.argv) > 1:
        base_url = sys.argv[1].rstrip("/")
    else:
        base_url = os.environ.get("VERCEL_URL", "").rstrip("/")
    if not base_url:
        print("Usage: python scripts/debug_500.py <VERCEL_URL>")
        sys.exit(1)

    url = base_url + "/api/prompt"
    print(f"\n{BOLD}── debug_500 ───────────────────────────────────────────{RESET}")
    print(f"  POST {url}")
    print(f"  Question: {QUESTION!r}")
    print(f"{'─' * 57}")

    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json={"question": QUESTION}, timeout=90)
    except requests.RequestException as exc:
        print(f"{RED}Request failed: {exc}{RESET}")
        sys.exit(1)

    elapsed = (time.perf_counter() - t0) * 1000
    status_color = GREEN if resp.status_code == 200 else RED
    print(f"\n  Status  : {status_color}{resp.status_code}{RESET}  ({elapsed:.0f} ms)")

    print(f"\n  {BOLD}Response headers:{RESET}")
    for k, v in resp.headers.items():
        print(f"    {k}: {v}")

    print(f"\n  {BOLD}Body (raw):{RESET}")
    print(f"  {resp.text[:2000]}")

    # Classify the error
    try:
        body = resp.json()
    except Exception:
        print(f"\n  {RED}Body is not valid JSON{RESET}")
        return

    print(f"\n  {BOLD}Body (parsed):{RESET}")
    print(json.dumps(body, indent=4))

    if resp.status_code == 200:
        response_text = (body.get("response") or "").strip()
        chunks = len(body.get("context", []))
        print(f"\n  {GREEN}200 OK — chunks={chunks}, response_len={len(response_text)}{RESET}")
        if not response_text:
            print(f"  {RED}WARNING: response is EMPTY — check LLM step in Vercel logs{RESET}")
        else:
            print(f"  {GREEN}LLM response preview: {response_text[:120]!r}{RESET}")
        return

    # Distinguish Vercel-internal 500 vs application error
    err = body.get("error", {})
    if isinstance(err, dict) and err.get("code") == "500" and "id" in err:
        print(f"\n  {RED}VERCEL-LEVEL 500{RESET} (Vercel caught an unhandled exception).")
        print(f"  Error ID: {err.get('id')}")
        print(f"\n  Next step: open Vercel dashboard → project → Functions → prompt")
        print(f"  and find the log lines for request ID {err.get('id')!r}.")
        print(f"  The last step=N OK line shows where execution stopped.")
    elif "error" in body:
        print(f"\n  {RED}APPLICATION {resp.status_code}{RESET}: {body['error']}")
        print(f"  This is the Python handler's own error — check step=N FAIL in logs.")
    else:
        print(f"\n  {RED}Unknown error format{RESET}")


if __name__ == "__main__":
    main()
