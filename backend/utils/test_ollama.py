"""
Run: python test_ollama_npredict.py
Tests whether higher num_predict lets Gemma4 finish thinking and produce visible output.
"""
import requests, time, os, json

BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# MODEL = os.getenv("OLLAMA_MODEL", "gemma3:27b")
# MODEL = os.getenv("OLLAMA_MODEL", "gemma4:26b")
# MODEL = os.getenv("OLLAMA_MODEL", "gemma4:31b")
MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")

HEADER_PROMPT = """
Generate a JSON object for a Call of Cthulhu scenario header.
Keep all values under 20 words. English only.

SEED: A lighthouse keeper reports that the light is attracting things from the sky instead of warning ships.
SETTING: Nautical Horror, golden age of ocean liners (1900s-1940s).

Return ONLY this JSON structure:
{
  "title": "",
  "era_and_setting": "",
  "atmosphere_notes": "",
  "inciting_hook": "",
  "core_mystery": "",
  "hidden_threat": ""
}
""".strip()


def call(num_predict, fmt=None, label=""):
    payload = {
        "model": MODEL, "prompt": HEADER_PROMPT, "stream": False,
        "options": {"temperature": 0.5, "num_predict": num_predict, "num_ctx": 8192},
    }
    if fmt:
        payload["format"] = fmt

    t0 = time.time()
    resp = requests.post(f"{BASE_URL}/api/generate", json=payload, timeout=180)
    data = resp.json()
    elapsed = time.time() - t0
    text = data.get("response", "")
    tokens = data.get("eval_count", 0)
    done_reason = data.get("done_reason", "?")
    status = "OK" if text.strip() else "EMPTY"

    print(f"  [{status}] {label}")
    print(f"    num_predict={num_predict}  format={fmt or 'none'}")
    print(f"    tokens_generated={tokens}  done_reason={done_reason}  visible_len={len(text)}  time={elapsed:.1f}s")
    if text.strip():
        print(f"    >>> {text.strip()[:400]}")
    else:
        print(f"    >>> (empty — all {tokens} tokens were invisible thinking)")
    print()


def main():
    print(f"Model: {MODEL}")
    print("=" * 70)

    # Group 1: no format=json, vary num_predict to find where thinking ends
    print("\n--- WITHOUT format=json (raw completion) ---")
    for np in [600, 900, 1500, 2000, 3000, 4096]:
        call(np, fmt=None, label=f"no-json / np={np}")

    # Group 2: with format=json for comparison
    print("\n--- WITH format=json ---")
    for np in [600, 900, 1500]:
        call(np, fmt="json", label=f"json / np={np}")

    # Group 3: num_predict=-1 (unlimited) without json
    print("\n--- UNLIMITED num_predict (-1) without json ---")
    call(-1, fmt=None, label="no-json / np=unlimited")

    print("=" * 70)
    print("Look for the num_predict threshold where visible output appears.")
    print("That's how many tokens Gemma4 burns on thinking before producing text.")


if __name__ == "__main__":
    main()