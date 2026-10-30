# Eval report — boolq

- **Timestamp**: 2026-10-05T12:37:29.909882+00:00
- **Model**: `Qwen/Qwen2.5-VL-7B-Instruct` @ `cc594898137f460bfe9f0759e9844b3ce807cfb5`
- **Config**: method=bernoulli; debias=reverse
- **Examples**: 3270

## Metrics

| metric | value |
|---|---|
| accuracy  | 0.6318 |
| macro_f1  | 0.4173 |
| ece (15)  | 0.0999 |
| brier     | 0.4723 |
| nll       | 0.6652 |

## Latency

| percentile | ms |
|---|---|
| p50  | 185.0 |
| p95  | 267.0 |
| mean | 190.5 |
