# Docker: what is possible and what is not

**GPU inference in a container on macOS is not possible.** Docker Desktop runs a
Linux VM; Metal/MLX cannot be passed through to it, and MLX has no Linux build.
So there is no Dockerfile that serves this engine with GPU acceleration on a Mac.

What is in this directory:

- `Dockerfile.bench`: a Linux image with the *benchmark client* only
  (`bench/agent_workload.py`, `bench/soak.py`). Use it to load-test a server that
  runs natively on the Mac host, from CI or another machine:

  ```bash
  docker build -f docker/Dockerfile.bench -t terrazzo-bench .
  docker run --rm --network host terrazzo-bench \
    --base http://127.0.0.1:8600 --model terrazzo --sessions 4 --turns 2 \
    --system-file /system.txt --max-tokens 64
  ```

- The supported "packaged" deployment for the server is the native path:
  `scripts/install.sh` + `packaging/mac/com.terrazzo.serve.plist` (launchd with
  keep-alive), or `scripts/serve_watchdog.sh` for a foreground supervisor.
