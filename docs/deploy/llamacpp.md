# llama.cpp Deployment Guide for RedSage (Apple Silicon / Metal)

High-throughput local inference for RedSage on Apple Silicon using llama.cpp with Metal GPU acceleration. This guide covers quantization, multi-instance serving behind an Nginx load balancer, and integration with the Claude Code security gate.

## Table of Contents

- [Quick Start](#quick-start)
- [Hardware Requirements](#hardware-requirements)
- [Build llama.cpp with Metal](#build-llamacpp-with-metal)
- [Quantize the Model](#quantize-the-model)
- [Single Instance](#single-instance)
- [Multi-Instance Serving](#multi-instance-serving)
- [Performance Benchmarks](#performance-benchmarks)
- [Security Gate Integration](#security-gate-integration)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# 1. Build llama.cpp with Metal
git clone https://github.com/ggml-org/llama.cpp /tmp/llama.cpp
cd /tmp/llama.cpp
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)

# 2. Quantize (bf16 -> Q4_K_M, 15GB -> 4.7GB)
./build/bin/llama-quantize RedSage-Qwen3-8B-DPO.gguf RedSage-Q4_K_M.gguf Q4_K_M

# 3. Serve
./build/bin/llama-server \
  --model RedSage-Q4_K_M.gguf \
  --port 8800 \
  --n-gpu-layers 999 \
  --flash-attn on \
  --parallel 4 \
  --ctx-size 8192

# 4. Query (OpenAI-compatible API)
curl http://localhost:8800/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"redsage","messages":[{"role":"user","content":"What is SSRF?"}]}'
```

---

## Hardware Requirements

| Config | RAM | GPU Cores | Model Format | Instances | Concurrent Slots |
|--------|-----|-----------|-------------|-----------|-----------------|
| Minimum | 16GB | 8+ | Q4_K_M (4.7GB) | 1 | 4 |
| Recommended | 32GB | 30+ | Q4_K_M (4.7GB) | 2 | 8 |
| Maximum throughput | 32GB | 30+ | Q4_K_M (4.7GB) | 4 | 16 |

**Tested on:** Apple M2 Max (32GB unified, 30 GPU cores, 12 CPU cores, Metal 4)

### Quantization Options

| Format | Size | Quality | Use Case |
|--------|------|---------|----------|
| bf16 (original) | 15.6GB | Best | Single instance, research |
| Q8_0 | ~8.5GB | Near-lossless | 2 instances on 32GB |
| Q6_K | ~6.6GB | Excellent | 3 instances on 32GB |
| **Q4_K_M** | **4.7GB** | **Good** | **4 instances on 32GB (recommended)** |
| Q4_K_S | ~4.4GB | Acceptable | Maximum instances |

Q4_K_M provides the best balance of quality and throughput for multi-instance serving.

---

## Build llama.cpp with Metal

```bash
git clone https://github.com/ggml-org/llama.cpp /tmp/llama.cpp
cd /tmp/llama.cpp

# Configure with Metal GPU support
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release

# Build (use all CPU cores)
cmake --build build -j$(sysctl -n hw.ncpu)

# Verify Metal support
./build/bin/llama-server --help 2>&1 | head -5
# Should show: ggml_metal_device_init: GPU name: Apple M2 Max
```

### Install Binaries (Optional)

```bash
INSTALL_DIR=/usr/local/bin  # or your preferred location
cp build/bin/llama-server build/bin/llama-quantize build/bin/llama-cli "$INSTALL_DIR/"
cp build/bin/*.dylib "$INSTALL_DIR/"
```

---

## Quantize the Model

### Download the GGUF

```bash
# From Hugging Face (if a GGUF is published)
huggingface-cli download RISys-Lab/RedSage-Qwen3-8B-DPO --include "*.gguf"

# Or convert from safetensors
pip install gguf
python convert_hf_to_gguf.py \
  --outfile RedSage-Qwen3-8B-DPO-bf16.gguf \
  --outtype bf16 \
  RISys-Lab/RedSage-Qwen3-8B-DPO
```

### Quantize to Q4_K_M

```bash
./build/bin/llama-quantize \
  RedSage-Qwen3-8B-DPO-bf16.gguf \
  RedSage-Qwen3-8B-DPO-Q4_K_M.gguf \
  Q4_K_M
```

Output:
```
llama_model_quantize_impl: model size  = 15623.18 MiB
llama_model_quantize_impl: quant size  =  4789.19 MiB
```

---

## Single Instance

```bash
DYLD_LIBRARY_PATH=/path/to/llama-cpp \
llama-server \
  --model RedSage-Qwen3-8B-DPO-Q4_K_M.gguf \
  --port 8800 \
  --host 127.0.0.1 \
  --n-gpu-layers 999 \
  --flash-attn on \
  --parallel 4 \
  --ctx-size 8192
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--parallel N` | 1 | Concurrent request slots |
| `--ctx-size N` | 2048 | Context window per slot |
| `--n-gpu-layers N` | 0 | Layers offloaded to GPU (999 = all) |
| `--flash-attn on` | auto | Flash attention for memory efficiency |
| `--port N` | 8080 | HTTP server port |

---

## Multi-Instance Serving

For maximum throughput, run multiple llama-server instances behind an Nginx load balancer. Each instance handles its own request queue with separate KV caches.

### Architecture

```
                     ┌─ llama-server :8801 (parallel=4, Metal GPU)
Nginx :8800 ─────── ├─ llama-server :8802 (parallel=4, Metal GPU)
(least_conn)         ├─ llama-server :8803 (parallel=4, Metal GPU)
                     └─ llama-server :8804 (parallel=4, Metal GPU)
```

### Install Nginx

```bash
brew install nginx  # macOS
# or
sudo apt install nginx  # Linux
```

### Launcher Script

Save as `start-redsage.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

LLAMA_SERVER="/path/to/llama-server"
MODEL="/path/to/RedSage-Qwen3-8B-DPO-Q4_K_M.gguf"
INSTANCES=4
PARALLEL=4
CTX_SIZE=8192
BASE_PORT=8801

for i in $(seq 1 "$INSTANCES"); do
    PORT=$(( BASE_PORT + i - 1 ))
    DYLD_LIBRARY_PATH="$(dirname "$LLAMA_SERVER")" "$LLAMA_SERVER" \
        --model "$MODEL" \
        --port "$PORT" \
        --host 127.0.0.1 \
        --parallel "$PARALLEL" \
        --ctx-size "$CTX_SIZE" \
        --n-gpu-layers 999 \
        --flash-attn on \
        > "/tmp/redsage-instance-${i}.log" 2>&1 &
    echo "Instance $i: port=$PORT pid=$!"
done

# Wait for all instances to be ready
for i in $(seq 1 "$INSTANCES"); do
    PORT=$(( BASE_PORT + i - 1 ))
    while ! curl -sf "http://127.0.0.1:${PORT}/health" &>/dev/null; do sleep 1; done
    echo "Instance $i ready on port $PORT"
done

# Start Nginx (see nginx config below)
nginx -c /path/to/nginx-redsage.conf
echo "Load balancer ready on port 8800"
```

### Nginx Configuration

Save as `nginx-redsage.conf`:

```nginx
worker_processes auto;
error_log /tmp/redsage-nginx-error.log warn;
pid /tmp/redsage-nginx.pid;

events { worker_connections 1024; }

http {
    access_log /tmp/redsage-nginx-access.log;

    upstream redsage {
        least_conn;
        server 127.0.0.1:8801;
        server 127.0.0.1:8802;
        server 127.0.0.1:8803;
        server 127.0.0.1:8804;
    }

    server {
        listen 8800;

        location / {
            proxy_pass http://redsage;
            proxy_set_header Host $host;
            proxy_read_timeout 300s;
            proxy_buffering off;  # Required for SSE streaming
        }
    }
}
```

### Stop Script

```bash
#!/usr/bin/env bash
pkill -f "llama-server.*RedSage" 2>/dev/null || true
nginx -s stop -c /path/to/nginx-redsage.conf 2>/dev/null || true
echo "All RedSage instances stopped"
```

---

## Performance Benchmarks

Tested on Apple M2 Max (32GB, 30 GPU cores) with Q4_K_M:

### Single Request Performance

| Metric | Value |
|--------|-------|
| Prompt processing | ~155 tok/s |
| Generation speed | ~51 tok/s |
| Time to first token | ~300ms |

### Throughput (4 Instances, 16 Slots)

| Concurrency | Aggregate tok/s | Avg Latency | Memory |
|-------------|----------------|-------------|--------|
| 1 | 51 tok/s | 1.3s | 6 GB |
| 4 | ~100 tok/s | 2.5s | 6 GB |
| 16 | ~28 tok/s | 22s | 24 GB |

### Memory Usage

| Instances | RSS Total | Headroom (32GB) |
|-----------|-----------|-----------------|
| 1 | ~6 GB | 26 GB |
| 2 | ~12 GB | 20 GB |
| 4 | ~24 GB | 8 GB |

---

## Security Gate Integration

RedSage can serve as a deep analysis backend for the [Claude Code Security](https://github.com/marc-shade/claude-code-security) framework's security gate pipeline.

### How It Works

The security gate runs 4 phases of content analysis:
1. **Injection detection** — Heuristic pattern matching
2. **Policy check** — Configurable forbidden patterns
3. **Threat intelligence** — URL/IP/payload detection
4. **RedSage deep analysis** — Contextual LLM reasoning (Phase 4)

Phase 4 only triggers when earlier phases produce HIGH+ severity findings. RedSage provides a contextual second opinion that can confirm threats or downgrade false positives.

### Enable Integration

```bash
pip install -e /path/to/claude-code-security
export CLAUDE_CODE_SECURITY_REDSAGE_ENABLED=true
export CLAUDE_CODE_SECURITY_REDSAGE_URL=http://localhost:8800/v1/chat/completions
```

### Standalone Security Analysis

```python
from claude_code_security.redsage_analyzer import analyze_content, extract_iocs

# Analyze flagged content
result = analyze_content("suspicious payload here", source="webfetch")
# Returns: {"verdict": "MALICIOUS", "confidence": 0.95, "category": "injection", ...}

# Extract IOCs from text
iocs = extract_iocs("outbound connection to 185.220.101.34 on port 9001")
# Returns: {"iocs": [{"type": "ip", "value": "185.220.101.34", ...}], ...}
```

---

## Troubleshooting

### Metal Not Detected

```
ggml_metal_device_init: GPU name: (empty)
```

Ensure you built with `-DGGML_METAL=ON` and are running on Apple Silicon (M1/M2/M3/M4).

### ccache Build Errors (Abort Trap 6)

```
dyld: Library not loaded: /opt/homebrew/opt/fmt/lib/libfmt.11.dylib
```

Fix: `brew reinstall ccache fmt`, then rebuild.

### Out of Memory

Reduce instances or parallel slots:
```bash
./start-redsage.sh 2 4   # 2 instances, 4 parallel slots each
```

### Slow Generation Under Concurrency

Metal GPU time-slices across instances. For maximum per-request speed at lower concurrency, use fewer instances with more parallel slots:
```bash
./start-redsage.sh 1 8   # 1 instance, 8 parallel slots
```

---

## Resources

- **llama.cpp:** https://github.com/ggml-org/llama.cpp
- **GGUF format:** https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- **Claude Code Security:** https://github.com/marc-shade/claude-code-security
- **RedSage Models:** https://huggingface.co/collections/RISys-Lab/redsage-models
