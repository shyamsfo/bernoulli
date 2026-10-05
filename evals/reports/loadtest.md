# Load test report

- **Timestamp**: 2026-10-05T23:34:59.934892+00:00
- **URL**: `http://127.0.0.1:8000`
- **Model**: `Qwen/Qwen2.5-VL-7B-Instruct` @ `cc594898137f460bfe9f0759e9844b3ce807cfb5`
- **Warmup requests**: 3

## Per-request latency (ms)

| questions/req | n | p50 | p95 | p99 | mean | stdev |
|---|---|---|---|---|---|---|
| 1 | 50 | 76.0 | 76.4 | 76.8 | 76.0 | 0.2 |
| 5 | 50 | 368.4 | 369.2 | 369.8 | 368.3 | 0.6 |
| 20 | 50 | 1465.9 | 1468.9 | 1470.1 | 1466.1 | 1.6 |
