# Exact distance verification for BB_144_12_12 via DistQLDPC (MaxSAT)

`Hx`/`Hz`/`Gx`/`Gz` are the matrices bundled with the external
[DistQLDPC](https://github.com/guluchen/DistQLDPC) repo's own sample data
(`data/matrices/BB_144_12_12_*`) -- IBM's published [[144,12,12]] bivariate
bicycle code from the ASC paper's Table II, not reconstructed here.

## Method

DistQLDPC (MaxCDCL-based MaxSAT solver, symplectic formulation) searches for
the minimum-weight nontrivial logical operator directly, reporting both a
lower bound (`d_lb`) and upper bound (`d_ub`) that converge to a certified
exact value when the MaxSAT solve terminates with status OPTIMAL.

`distqldpc_maxsat_log.txt`: full solver log, `-cpu-lim=280 -v`. Took ~65s
wall time -- noticeably more than the n<=108 codes' ~5-8s each, but still
tractable (unlike n=288, which did not converge within 30 minutes -- see
that code's own attempt, reported separately).

## Result

```
c Hx: 72 x 144, Hz: 72 x 144, Gx: 12 x 144, Gz: 12 x 144, logicals: 24
c d_lb: 12
c d_ub: 12
c d  : 12
o 12
```

**d = 12, PROVEN EXACT** (MaxSAT status: OPTIMAL) -- matches the paper's published `d=12`, now
independently re-derived rather than just cited.

So `BB_144_12_12`: **[[144, 12, 12]]**, exactly.
