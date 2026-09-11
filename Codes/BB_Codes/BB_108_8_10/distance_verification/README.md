# Exact distance verification for BB_108_8_10 via DistQLDPC (MaxSAT)

`Hx`/`Hz`/`Gx`/`Gz` are the matrices bundled with the external
[DistQLDPC](https://github.com/guluchen/DistQLDPC) repo's own sample data
(`data/matrices/BB_108_8_10_*`) -- IBM's published [[108,8,10]] bivariate
bicycle code from the ASC paper's Table II, not reconstructed here.

## Method

DistQLDPC (MaxCDCL-based MaxSAT solver, symplectic formulation) searches for
the minimum-weight nontrivial logical operator directly, reporting both a
lower bound (`d_lb`) and upper bound (`d_ub`) that converge to a certified
exact value when the MaxSAT solve terminates with status OPTIMAL.

`distqldpc_maxsat_log.txt`: full solver log, `-cpu-lim=280 -v`.

## Result

```
c Hx: 54 x 108, Hz: 54 x 108, Gx: 8 x 108, Gz: 8 x 108, logicals: 16
c d_lb: 10
c d_ub: 10
c d  : 10
o 10
```

**d = 10, PROVEN EXACT** (MaxSAT status: OPTIMAL) -- matches the paper's published `d=10`, now
independently re-derived rather than just cited.

So `BB_108_8_10`: **[[108, 8, 10]]**, exactly.
