import time
import json
import statistics as st
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
import src.LocalASC.scheduling.localASC_scheduler as L

def profile_schedule_generation(
    code_name: str,
    Hx: np.ndarray,
    Hz: np.ndarray,
    policy: str,
    output_file: Path,
    runs: int = 10,
    tlim: int = 1800,       
    tlim_step: int = 60,     
    tlim_hybrid: int = 10,    
    files_metadata: list = []
) -> dict:
    """
    Profiles the schedule generation algorithm for given parity check matrices.
    """
    w = int(max(Hx.sum(1).max(), Hz.sum(1).max()))
    Delta = int((Hx.sum(0) + Hz.sum(0)).max())
    D = max(w, Delta)
    
    edges = [("X", r, int(q)) for r in range(Hx.shape[0]) for q in np.nonzero(Hx[r])[0]] + \
            [("Z", r, int(q)) for r in range(Hz.shape[0]) for q in np.nonzero(Hz[r])[0]]
            
    T, A, statuses, tried, paths = [], [], [], None, []
    
    for i in range(runs):
        t0 = time.perf_counter()
        autos = L.automorphisms(Hx, Hz)
        t1 = time.perf_counter()
        
        if policy in ("ladder", "hybrid"):
            if policy == "ladder":
                gens, orb, norb, status, vals, tr = L.ladder(Hx, Hz, edges, autos, D, skip=None, above=0, tlim_step=tlim_step)
            else:
                gens, orb, norb, status, vals, tr = L.hybrid(
                    Hx, Hz, edges, autos, D, tlim_hybrid=tlim_hybrid, tlim_step=tlim_step, tlim_max=tlim
                )
                
            tried = tried or tr
            if policy == "hybrid":
                paths.append(tr[-1][0])
        else:
            gens = L.pick(autos, policy)
            orb, norb = L.orbit_map(edges, gens)
            inj, why = L.is_locally_injective(edges, orb)
            assert inj, f"{code_name}: H not well-formed ({why})"
            status, _, vals = L.solve_equivariant(Hx, Hz, orb, norb, D, tlim=tlim)
            
        t2 = time.perf_counter()
        T.append(t2 - t0)
        A.append(t1 - t0)
        statuses.append(status)
        
        if i == 0 and vals is not None:
            _, _, conf, bad, pairs, _, ne = L.lift_validate(Hx, Hz, orb, vals)
            lift = f"{ne} edges, {conf} conflicts, {bad}/{pairs} parity violations"
        elif i == 0:
            lift = "no solution"
            
        print(f"  {code_name} run {i + 1}: {status} {(t2 - t0) * 1e3:.1f} ms", flush=True)
        if status != "OPTIMAL":
            break

    H = len(L._expand_closure(gens)) if gens else 1
    
    sd_ms = round(st.pstdev(T) * 1e3, 1) if len(T) > 1 else 0.0
    
    rec = {
        "code": code_name,
        "n": int(Hx.shape[1]),
        "mean_ms": round(st.mean(T) * 1e3, 1),
        "w": w,
        "Delta": Delta,
        "D": D,
        "aut": len(autos),
        "policy": policy,
        "H": H,
        "orbits": norb,
        "edges": len(edges),
        "statuses": sorted(set(statuses)),
        "lift": lift,
        "times_ms": [round(t * 1e3, 1) for t in T],
        "sd_ms": sd_ms,
        "aut_mean_ms": round(st.mean(A) * 1e3, 1),
        "date": time.strftime("%Y-%m-%d"),
        "files": files_metadata or []
    }
    
    if tried is not None:
        rec["tried"] = tried
    if paths:
        rec["paths"] = {p: paths.count(p) for p in set(paths)}
        
    print(f"{code_name}: D={D} |Aut|={len(autos)} {policy} |H|={H} orbits={norb}/{len(edges)} {rec['statuses']} "
          f"mean {rec['mean_ms']} ms (sd {rec['sd_ms']}; Aut {rec['aut_mean_ms']} ms), lift: {lift}")
          
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "a") as f:
            f.write(json.dumps(rec) + "\n")
            
    return rec