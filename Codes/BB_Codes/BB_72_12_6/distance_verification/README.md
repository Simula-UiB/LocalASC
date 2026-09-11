# Exact distance verification for BB_72_12_6 via DistQLDPC (MaxSAT)

`Hx`/`Hz`/`Gx`/`Gz` are the matrices bundled with the external
[DistQLDPC](https://github.com/guluchen/DistQLDPC) repo's own sample data
(`data/matrices/BB_72_12_6_*`) -- these are IBM's published [[72,12,6]]
bivariate bicycle code from the ASC paper's Table II, not reconstructed here.

## Method

DistQLDPC (MaxCDCL-based MaxSAT solver, symplectic formulation) searches for
the minimum-weight nontrivial logical operator directly, reporting both a
lower bound (`d_lb`) and upper bound (`d_ub`) that converge to a certified
exact value when the MaxSAT solve terminates with status OPTIMAL.

`distqldpc_maxsat_log.txt`: full solver log, `-cpu-lim=280 -v`.

## Result

```
c Hx: 36 x 72, Hz: 36 x 72, Gx: 12 x 72, Gz: 12 x 72, logicals: 24
c d_lb: 6
c d_ub: 6
c d  : 6
o 6
```

**d = 6, PROVEN EXACT** (MaxSAT status: OPTIMAL) -- matches the paper's published `d=6`, now
independently re-derived rather than just cited.

So `BB_72_12_6`: **[[72, 12, 6]]**, exactly.
