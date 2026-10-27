"""Watch a running Terrazzo server; dump thread stacks if /health stops answering.

Usage:
    python scripts/watch_health.py --port 8991 --pid <server-pid> --log /tmp/server.log
The server must be started with TERRAZZO_STACK_DUMP=1 so SIGUSR1 dumps every
thread stack to stderr (which the harness redirects to its log).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import time
import urllib.request


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--log", default=None)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--failures", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    fails = 0
    start = time.time()
    while time.time() - start < args.timeout:
        time.sleep(args.interval)
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as r:
                if r.status == 200:
                    fails = 0
                    continue
        except Exception:
            pass
        fails += 1
        print(f"[watch] health miss {fails}/{args.failures}", flush=True)
        if fails < args.failures:
            continue
        print(f"[watch] HANG suspected on {base}; sending SIGUSR1 to {args.pid}", flush=True)
        try:
            with urllib.request.urlopen(base + "/stats", timeout=2) as r:
                print("[watch] stats:", r.read().decode()[:1000], flush=True)
        except Exception:
            pass
        try:
            os.kill(args.pid, signal.SIGUSR1)
            print("[watch] stack dump requested; see the server log", flush=True)
        except Exception as exc:
            print("[watch] SIGUSR1 failed:", exc, flush=True)
        if args.log:
            print(f"[watch] server log: {args.log}", flush=True)
        return 1
    print("[watch] timeout without hang", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
