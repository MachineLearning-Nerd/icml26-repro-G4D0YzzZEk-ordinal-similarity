# Claim 1 — Ordinal TSI and QSI


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_b5ee832a2468", "created_at": "2026-07-17T06:40:19+00:00", "title": "Claim and result"}
-->
## Claim
Triplet Similarity Index (TSI) and Quadruplet Similarity Index (QSI) measure representation alignment through agreement of ordinal distance relationships.

## Independent audit
Forty no-tie representation pairs were checked with literal ordered-triplet and ordered-quadruplet enumeration. A separate rank implementation matched TSI exactly (maximum error 0.0). Rotation, scaling, and translation preserved every ordinal relation: TSI = 1 and QSI = 1 in all 40 cases.

## Large sampled path
At n=10,000 and d=512 with 20,000 comparisons per metric, aligned representations scored TSI 0.99995 and QSI 1.0; a row-permuted control scored 0.50610 and 0.49995.
