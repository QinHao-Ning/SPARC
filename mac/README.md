# MAC solver for SPARC

This directory contains the MAC runtime used by SPARC: the MAC solver,
Frank-Wolfe optimization, rounding, graph utilities, and Fiedler-vector
backends. The offline entry point is `sim2mac_fixed.py`.

Install from the SPARC repository root after its dependencies:

```bash
python -m pip install --no-deps -e ./mac
```

The default `tracemin_lu` backend uses NumPy, SciPy, and NetworkX.
`tracemin_cholesky` is optional and requires SuiteSparse and
`scikit-sparse`; it is not needed for the default path.

Upstream robotics datasets, examples, tests, and alternative greedy
solvers are omitted from this publication subset. The retained numerical
solver and preprocessing code are unchanged.

Original implementation: https://github.com/MarineRoboticsGroup/mac

Copyright and MIT terms are preserved in [LICENSE](LICENSE).
