# Conclusion


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_d04ff9b7a1be", "created_at": "2026-07-17T06:40:21+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-17T06:40:22+00:00"}
-->
## Result

Both claim-level results are reproduced. The literal ordinal definitions and an independent rank formulation agree exactly; perfect TSI is equivalent to all-scale MutualNN agreement in 960 no-tie cases. The public approximate path also separates aligned from permuted representations at n=10,000,d=512.

## Scope & cost

| | This reproduction | Full replication |
|---|---|---|
| Scope | Definitions, Corollary 1, official sanity suite, n=10,000 sampled path | All CIFAR-10 and CLIP training studies |
| Hardware | CPU, 4 vCPU | GPU training and feature extraction |
| Time | under one minute | multi-epoch model-training campaigns |
| Cost | local CPU | GPU compute |
| Outcome | Both scored algorithmic/theorem claims verified | Training-table measurements not claimed |
