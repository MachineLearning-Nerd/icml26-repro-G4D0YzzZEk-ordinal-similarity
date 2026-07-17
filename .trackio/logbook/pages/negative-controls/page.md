# Negative controls


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_b04465fa37fe", "created_at": "2026-07-17T06:40:21+00:00", "title": "Preconditions and false-positive controls"}
-->
1. A random row permutation preserves the marginal data but destroys correspondence; sampled TSI and QSI collapse to about 0.5.
2. A tied-distance point cloud is rejected before applying Corollary 1 because nearest-neighbor sets are not unique.
3. The official code's own complete sanity suite is run alongside the independent tests, so a mismatch cannot be hidden by a single implementation.


---
<!-- trackio-cell
{"type": "code", "id": "cell_af84993f9c3d", "created_at": "2026-07-17T06:40:51+00:00", "title": "Independent audit tests", "command": ["python", "-m", "pytest", "-q", "repro/tests"], "exit_code": 0, "duration_s": 0.292}
-->
````bash
$ python -m pytest -q repro/tests
````

exit 0 · 0.3s


````output
....                                                                     [100%]
4 passed in 0.07s

````
