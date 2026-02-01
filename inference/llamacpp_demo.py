#!/usr/bin/env python3
"""
RedSage llama.cpp Client Demo

Demonstrates interacting with RedSage served via llama-server (llama.cpp)
using the OpenAI-compatible API. Works with both single-instance and
multi-instance (Nginx LB) deployments.

Prerequisites:
    - llama-server running with RedSage Q4_K_M GGUF
    - openai Python package installed (`pip install openai`)

Usage:
    python llamacpp_demo.py
    python llamacpp_demo.py --base-url http://localhost:8800/v1
    python llamacpp_demo.py --concurrent 16  # Throughput test
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

DEFAULT_BASE_URL = "http://localhost:8800"
DEFAULT_SYSTEM_PROMPT = "You are RedSage, a helpful cybersecurity assistant."


def query(
    base_url: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.2,
    stream: bool = False,
) -> dict:
    """Send a chat completion request to llama-server."""
    payload = json.dumps({
        "model": "redsage",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def health_check(base_url: str) -> bool:
    """Check if the server is healthy."""
    try:
        req = urllib.request.Request(f"{base_url}/health", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("status") == "ok"
    except Exception:
        return False


def run_examples(base_url: str, max_tokens: int, temperature: float):
    """Run example cybersecurity queries."""
    examples = [
        ("SSRF Mitigation", "List three SSRF mitigations."),
        ("CVSS Vector", "Explain AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H."),
        ("SQL Injection", "What are best practices to prevent SQL injection?"),
        ("Zero-Day Response", "Outline an incident response plan for a zero-day exploit."),
    ]

    print("\n" + "=" * 70)
    print("RedSage llama.cpp Demo - Example Queries")
    print("=" * 70)

    for i, (title, question) in enumerate(examples, 1):
        print(f"\n[{i}/{len(examples)}] {title}")
        print("-" * 70)
        print(f"User: {question}\n")

        try:
            start = time.time()
            result = query(
                base_url,
                [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            elapsed = time.time() - start

            content = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})
            timings = result.get("timings", {})

            print(f"RedSage: {content}\n")
            print(
                f"  [{usage.get('completion_tokens', '?')} tokens, "
                f"{timings.get('predicted_per_second', 0):.1f} tok/s, "
                f"{elapsed:.1f}s]"
            )
        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 70)


def throughput_test(base_url: str, concurrency: int, max_tokens: int):
    """Run a concurrent throughput benchmark."""
    print(f"\n{'=' * 70}")
    print(f"Throughput Test: {concurrency} concurrent requests")
    print(f"{'=' * 70}\n")

    def single_request(idx: int) -> dict:
        start = time.time()
        result = query(
            base_url,
            [{"role": "user", "content": "What is a buffer overflow? One sentence."}],
            max_tokens=64,
            temperature=0.1,
        )
        elapsed = time.time() - start
        tokens = result["usage"]["completion_tokens"]
        return {"idx": idx, "tokens": tokens, "elapsed": elapsed}

    start_all = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(single_request, i) for i in range(concurrency)]
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                print(f"  Request failed: {e}")

    total_time = time.time() - start_all
    total_tokens = sum(r["tokens"] for r in results)
    avg_latency = sum(r["elapsed"] for r in results) / len(results) if results else 0
    min_lat = min(r["elapsed"] for r in results) if results else 0
    max_lat = max(r["elapsed"] for r in results) if results else 0

    print(f"Results ({len(results)}/{concurrency} succeeded):")
    print(f"  Total tokens:       {total_tokens}")
    print(f"  Wall clock time:    {total_time:.1f}s")
    print(f"  Aggregate tok/s:    {total_tokens / total_time:.1f}")
    print(f"  Requests/sec:       {len(results) / total_time:.2f}")
    print(f"  Latency avg/min/max: {avg_latency:.1f}s / {min_lat:.1f}s / {max_lat:.1f}s")
    print(f"{'=' * 70}\n")


def interactive_mode(base_url: str, max_tokens: int, temperature: float):
    """Interactive chat session."""
    print(f"\n{'=' * 70}")
    print("RedSage Interactive Chat (llama.cpp)")
    print("Type 'quit' to exit, '/bench N' for throughput test")
    print(f"{'=' * 70}\n")

    messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break
        if user_input.startswith("/bench"):
            parts = user_input.split()
            n = int(parts[1]) if len(parts) > 1 else 16
            throughput_test(base_url, n, max_tokens)
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            result = query(base_url, messages, max_tokens=max_tokens, temperature=temperature)
            content = result["choices"][0]["message"]["content"]
            messages.append({"role": "assistant", "content": content})
            timings = result.get("timings", {})
            print(f"\nRedSage: {content}")
            print(f"  [{timings.get('predicted_per_second', 0):.1f} tok/s]\n")
        except Exception as e:
            messages.pop()  # Remove failed user message
            print(f"\nError: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="RedSage llama.cpp client demo")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Server base URL")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--concurrent", type=int, default=0,
                        help="Run throughput test with N concurrent requests")
    parser.add_argument("--examples-only", action="store_true",
                        help="Run examples then exit (no interactive mode)")
    args = parser.parse_args()

    print(f"Connecting to: {args.base_url}")
    if not health_check(args.base_url):
        print(f"ERROR: Server not reachable at {args.base_url}")
        print("Start it with: ./start-redsage.sh")
        sys.exit(1)
    print("Server: HEALTHY\n")

    if args.concurrent > 0:
        throughput_test(args.base_url, args.concurrent, args.max_tokens)
        return

    run_examples(args.base_url, args.max_tokens, args.temperature)

    if not args.examples_only:
        interactive_mode(args.base_url, args.max_tokens, args.temperature)


if __name__ == "__main__":
    main()
