# Negative controls


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_b04465fa37fe", "created_at": "2026-07-17T06:40:21+00:00", "title": "Preconditions and false-positive controls"}
-->
1. The authors' initial-versus-final CIFAR features give imperfect TSI and
   imperfect all-scale MutualNN, testing the reverse direction on real primary
   data.
2. A 2% CIFAR outlier injection stays imperfect/imperfect.
3. A cluster translation keeps fixed-k MutualNN equal to one at k=399 while
   TSI and all-scale MutualNN are both imperfect, exposing the fixed-k trap.
4. A tied grid is rejected before Corollary 1; deterministic no-tie jitter plus
   a point perturbation then gives imperfect/imperfect.
5. An independent verifier does not import the main exact-audit module.


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
