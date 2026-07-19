# Conclusion


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_d04ff9b7a1be", "created_at": "2026-07-17T06:40:21+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-17T06:40:22+00:00"}
-->
## Result

Both claim-level results are reproduced. The Claim 2 paper-scale repair uses
exactly 10 approaches and no more: 362,880 exhaustive strict orders, authors'
CIFAR features, three real datasets, and adversarial boundary controls. All
routes pass and a separate fail-closed verifier passes.

## Scope & cost

| | This reproduction | Full replication |
|---|---|---|
| Scope | Definitions, Corollary 1, exact n=2,048 CIFAR audits, official sanity suite | All CIFAR-10 and CLIP training studies |
| Hardware | CPU, 4 vCPU | GPU training and feature extraction |
| Time | about one minute | multi-epoch model-training campaigns |
| Cost | local CPU | GPU compute |
| Outcome | Both scored algorithmic/theorem claims verified | Training-table measurements not claimed |
