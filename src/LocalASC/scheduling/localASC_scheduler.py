import networkx as nx
import numpy as np
from ortools.sat.python import cp_model
from networkx.algorithms.isomorphism import GraphMatcher, categorical_node_match
import pynauty
import time
import itertools
###################
# Orbit functions #
###################

def orbit_map(edges, gens):
    """
    Compute the orbit map pi : E(T) -> E(T)/H for the subgroup H = <gens>
    gens only needs to contain a generating set.

    Union find implementation over the set of edges.
    
    Returns (orb, norb): 
    `orb` maps each edge to an integer orbit id in range(norb)
    `norb` is the number of distinct H-orbits
    """
    #Initialize edge dictionary
    parent = {e: e for e in edges}

    #Find function for union find, returns representative edge for an edge orbit
    def find(x):
        # if parent[x] == x:
           # return x
        #find(parent[x])
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x

    #Union part, combines two sets by putting them under the same representative class
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb

    #Loop puts edges into orbits of the acvtion by group generators
    for g in gens:
        for e in edges: union(e, edge_action(g, e))

    # Relabel each orbit's arbitrary union-find in first-seen order, so the ids can index directly into a CP-SAT variable array.
    reps = {}
    out = {}
    for e in edges:
        r = find(e)
        if r not in reps: reps[r] = len(reps)
        out[e] = reps[r]
    return out, len(reps)

def is_locally_injective(edges, orb):
    """
    Check the precondition that pi is injective on the incident edges of every check and every qubit.
    If two DISTINCT edges at the same node land in the same orbit, no consistent lift can satisfy C1 (or C2) there, no matter what tau is chosen.

    Returns (True, None) if well-formed, or (False, description) where
    `description` names one witness violation (the first one found).
    """
    #Initialize qubit dictionaries
    by_check = {}
    by_qubit = {}
    
    for e in edges:
        #Tanner graphs are bipartiete so edges connect a check to a data qubit
        typ, row, q = e
        #edge e corresponds to check (typ,row) and qubit q
        by_check.setdefault((typ, row), []).append(e)
        by_qubit.setdefault(q, []).append(e)

    for key, es in by_check.items():
        #Obtain the orbits seen by edges incident to each check
        orbits_seen = [orb[e] for e in es]
        #If the list of seen orbits contains a double occupancy/ is not injective, then the condition below is true
        if len(set(orbits_seen)) != len(orbits_seen):
            return False, f"check {key}: edges {es} collapse to only {len(set(orbits_seen))} distinct orbits"

    #Same logic as above
    for q, es in by_qubit.items():
        orbits_seen = [orb[e] for e in es]
        if len(set(orbits_seen)) != len(orbits_seen):
            return False, f"qubit {q}: edges {es} collapse to only {len(set(orbits_seen))} distinct orbits"

    return True, None

######################
# Graph construction #
######################

def tanner(Hx, Hz):
    """
    Build the Tanner graph T of a CSS code from its parity-check matrices. 
    Vertices are labelled with a 't' attribute in {'X','Z','q'} denoting vertex type.
    Automorphisms are required to preserve vertex type. 
    An edge (X,r)--(q,c) exists iff Hx[r,c]==1.
    Same edge set E(T) used everywhere else in this module.
    Returns T as a networkx.Graph.
    """
    G = nx.Graph()
    #Add X, Z and variable nodes
    for r in range(Hx.shape[0]): G.add_node(("X", r), t="X")
    for r in range(Hz.shape[0]): G.add_node(("Z", r), t="Z")
    for q in range(Hx.shape[1]): G.add_node(("q", q), t="q")
    
    #Loop over X checks and connect variable nodes involved
    for r in range(Hx.shape[0]):
        for q in np.nonzero(Hx[r])[0]: G.add_edge(("X", r), ("q", int(q)))

    #Same as above but with Z checks
    for r in range(Hz.shape[0]):
        for q in np.nonzero(Hz[r])[0]: G.add_edge(("Z", r), ("q", int(q)))

    return G

def _tanner_to_pynauty(Hx, Hz):
    """
    Build the same Tanner graph as tanner() above, but as a
    pynauty.Graph with vertex coloring encoding the X/Z/qubit type constraint.
    """

    #Extract number of checks and data qubits
    mx, n = Hx.shape
    mz, _ = Hz.shape

    #Create offsets to label nodes
    X_off, Z_off, Q_off = 0, mx, mx + mz

    #Total number of nodes (also total number of qubits)
    N = mx + mz + n

    #Adjacency dictionary
    adj = {v: [] for v in range(N)}

    #Loop over all X checks and add neighboring nodes to the corresponding adjacency lists
    for r in range(mx):
        for q in np.nonzero(Hx[r])[0]:
            adj[X_off + r].append(Q_off + int(q))
            adj[Q_off + int(q)].append(X_off + r)

    #Loop over all Z checks and add neighboring nodes to the corresponding adjacency lists
    for r in range(mz):
        for q in np.nonzero(Hz[r])[0]:
            adj[Z_off + r].append(Q_off + int(q))
            adj[Q_off + int(q)].append(Z_off + r)

    #Add colors to nodes depending on their type
    coloring = [set(range(X_off, X_off + mx)), set(range(Z_off, Z_off + mz)),
                set(range(Q_off, Q_off + n))]

    #Create the pynauty graph
    g = pynauty.Graph(N, directed=False, adjacency_dict=adj, vertex_coloring=coloring)

    #Return graph but also number of X, Z checks, number of physical qubits, and each offset for node labeling.
    return g, mx, mz, n, X_off, Z_off, Q_off

#######################
# Graph Automorphisms #
#######################

def _automorphisms_vf2(Hx, Hz, cap=2000):
    """
    VF2 (Cordella et al. 2004) algorithm, implemented in NetworkX's GraphMatcher.
    Yields a list of node-type preserving automorphisms on the graph. List has length cap.
    """
    #Initialize the tanner graph
    G = tanner(Hx, Hz)
    #Generator of automorphisms that preserve node type using node_match. 
    gm = GraphMatcher(G, G, node_match=categorical_node_match("t", None))

    return [dict(iso) for _, iso in zip(range(cap), gm.isomorphisms_iter())]

def _expand_closure(gens, cap=2000):
    """
    Given a set of generators 'gens', compute full generated group by BFS
    closure: start from identity, repeatedly compose with each generator,
    until no new elements appear or `cap` is reached. Combined with . 
    """
    # Composition function for automorphisms, returns the automorphism resulting from g \circ h
    def compose(g, h):
        return {k: g[h[k]] for k in h}

    #In case of empty generator list
    if not gens:
        return [{}]

    
    keys = set()
    #Construct the set of all keys from the generators.
    for g in gens:
        keys |= set(g.keys())
    
    #Base case for automorphisms, this is the identity
    identity = {k: k for k in keys}

    #Initialize BFS: we store seen automorphisms in seen and initialize frontier at the identity 
    seen = {tuple(sorted(identity.items())): identity}
    frontier = [identity]

    #BFS loop
    while frontier and len(seen) < cap:
        new_frontier = []
        #For every automorphism in the frontier we compose it with every generator and store new automorphisms until we have seen more than cap automorphisms or no new elements appear. 
        for x in frontier:
            for g in gens:
                y = compose(g, x)
                key = tuple(sorted(y.items()))
                if key not in seen:
                    seen[key] = y
                    new_frontier.append(y)
                    if len(seen) >= cap:
                        break
            if len(seen) >= cap:
                break
        frontier = new_frontier
    return list(seen.values())

def automorphisms(Hx, Hz, cap=2000):
    """
    Compute Aut(T). First tries nauty (McKay et al.) and if there is a problem with the package runs netwrokx GraphMatching.
    nauty is faster but automorphism computation is the same regardless.
    """
    try:
        #Obatin nauty version of Tanner Graph
        g, mx, mz, n, X_off, Z_off, Q_off = _tanner_to_pynauty(Hx, Hz)
        
        #Compute raw generators of automorphism group
        gens_raw, _, _, _, _ = pynauty.autgrp(g)

        #Function to map vertex index to vertex key (Including type)
        def vertex_to_key(v):
            if v < Z_off: return ("X", v - X_off)
            elif v < Q_off: return ("Z", v - Z_off)
            else: return ("q", v - Q_off)

        gens = []
        #Loop over all generetars and keep only automorphisms that preserve node type
        for perm in gens_raw:
            iso = {}
            for v in range(mx + mz + n):
                iso[vertex_to_key(v)] = vertex_to_key(perm[v])
            gens.append(iso)
        #Finally call _expand_closure to obtain the full node type preserving automorphism group as required by LocalASC
        return _expand_closure(gens, cap=cap)
    
    except Exception:
        ##In case something goes wrong with pynauty package 
        return _automorphisms_vf2(Hx, Hz, cap=cap)

def edge_action(iso, e):
    """
    Apply one automorphism `iso` to one edge e = (typ, check, qubit) of E(T).
    typ is X or Z depending on which PCM the edge comes from, check is the check index and qubit is the data qubit index.
    """
    typ, check, q = e
    #Returns the transformed edge
    return (typ, iso[(typ, check)][1], iso[("q", q)][1])

def order_of(iso):
    """
    Determine the order of automorphisms
    """
    p = {k[1]: v[1] for k, v in iso.items() if k[0] == "q"}
    o = 1
    cur = dict(p)
    while any(k != v for k, v in cur.items()):
        cur = {k: p[v] for k, v in cur.items()}
        o += 1
    return o

def pick(autos, policy):
    """
    Given a policy, returns a list of automorphisms, either all non-trivial automorphisms ("full") or all odd order automorphisms ("odd").
    In case a wrong policy name is given, returns the automorphism with the maximal order or an empty list if there are none.
    """
    nontriv = [a for a in autos if order_of(a) > 1]
    odd = [a for a in nontriv if order_of(a) % 2 == 1]
    if policy == "full":
        return nontriv
    if policy == "odd":
        return odd
    return [max(odd, key=order_of)] if odd else []   

def subgroup_key(elems):
    """
    Canonical, order-independent identity of a subgroup: the frozenset of its elements' own
    canonical keys, so two generator sets that produce the SAME subgroup dedupe to the same key.
    """
    return frozenset(tuple(sorted(e.items())) for e in elems)

def enumerate_subgroups(autos, rank=2, candidates=40, cap=2000):
    """
    Returns a dict: subgroup_key -> (generators, |H|), covering the trivial
    group, every cyclic subgroup, and rank <= `rank` combinations of a reduced
    candidate list, plus the full group Aut(T).
    """
    #Initialize the dictionary of found subgroups
    found = {}
    
    #Add the trivial subgroup:
    triv = _expand_closure([], cap=cap)
    found[subgroup_key(triv)] = ([], len(triv))

    #List all non identity automorphisms
    non_identity = [a for a in autos if order_of(a) > 1]

    #Generate the cyclic subgroups using the non identity elements
    for a in non_identity:
        elems = _expand_closure([a], cap=cap)
        key = subgroup_key(elems)
        if key not in found: found[key] = ([a], len(elems))

    # Reduced candidate list for larger subgroups, take the largest order generators (They tend to produce most useful subgroups)
    by_size = sorted((v for v in found.values() if v[0]), key=lambda v: -v[1])
    cand = [v[0][0] for v in by_size][:candidates]

    #Take ´cand´ choose r different combinations of generators and save them if they are new
    for r in range(2, max(rank, 1) + 1):
        for combo in itertools.combinations(cand, r):
            elems = _expand_closure(list(combo), cap=cap)
            key = subgroup_key(elems)
            if key not in found: found[key] = (list(combo), len(elems))

    #Create the subgroup generated by every non_identity automorphism we give and add it to the list of subgroups if we haven't seen it
    full_gens = non_identity[:max(rank, 1)] or []
    full_elems = _expand_closure(full_gens, cap=cap) if full_gens else list(autos)
    key = subgroup_key(full_elems)
    if key not in found: found[key] = (full_gens, len(full_elems))

    # Generate the whole group in case the closure of `full_gens` didn't reach it
    if found[key][1] != len(autos) and len(autos) > 1:
        key = subgroup_key(autos)
        found[key] = (non_identity, len(autos))

    return found


######################
# LocalASC functions #
######################

def cycle_aligned_domain(edges, orb, norb, D):
    """
    Returns orbit_domain for the staggered circuit.
    """
    dom = [None] * norb
    for e in edges:
        want = (1, D - 1) if e[0] == "X" else (0, D - 2)
        o = orb[e]
        if dom[o] is None: dom[o] = want
        elif dom[o] != want: raise ValueError(f"orbit {o} mixes X and Z edges")
    return dom

def solve_equivariant(Hx, Hz, orb, norb, D, tlim=120, orbit_domain=None, seed=None, exclude=None):
    """
    (Custom amount of workers for parallelization in the CP-solver)
    
    First step of LocalASC: Orbit model solving (line 11 in paper pseudo-code). Solve the H-orbit model at fixed
    depth D via CP-SAT. One integer variable tau_o in {0,...,D-1} per orbit.

    Solve the model using CP-SAT and conditions:
      - C1: TICKs on each check pairwise distinct
      - C2: TICKs on each qubit pairwise distinct
      - C3: commutation parity, for every overlapping X/Z.

    Returns (status_name, elapsed_seconds, values_or_None):
    status_name is CP-SAT's own StatusName ("OPTIMAL","FEASIBLE","INFEASIBLE",
    "UNKNOWN") in the normal case, or "VACUOUS" if the found solution is not a feasible orbit assignment.
    values_or_None is the list of solved tau_o values (indexed to match V) when a solution was found, else None.
    """
    #Initialize edges in the Tanner Graph of the code
    edges = [("X", r, int(q)) for r in range(Hx.shape[0]) for q in np.nonzero(Hx[r])[0]] + \
            [("Z", r, int(q)) for r in range(Hz.shape[0]) for q in np.nonzero(Hz[r])[0]]
    
    ok, why = is_locally_injective(edges, orb)

    #Checks if the given orbit assignment is correct
    if not ok:
        return "VACUOUS", 0.0, None

    #Initialize CP-SAT model
    md = cp_model.CpModel()

    #orbit_domain is a list of (lo,hi) ticks for every orbit, optional. When used helps obtain staggered circuits. 
    if orbit_domain is None:
        orbit_domain = [(0, D - 1)] * norb
    #Adds the constraint that each orbit's TICKs must be within the interval (lo,hi), this doesn't change the initial optimization problem, 
    #it adds the possibility to restrict when X and Z type checks can happen for staggered circuits.
    V = [md.NewIntVar(lo, hi, "") for (lo, hi) in orbit_domain]

    # tX[r] / tZ[r]: for check row r, a dict {qubit -> orbit variable} to save each check's incident edges.
    tX = {}; tZ = {}
    for r in range(Hx.shape[0]):
        for q in np.nonzero(Hx[r])[0]: tX.setdefault(r, {})[int(q)] = V[orb[("X", r, int(q))]]
    for r in range(Hz.shape[0]):
        for q in np.nonzero(Hz[r])[0]: tZ.setdefault(r, {})[int(q)] = V[orb[("Z", r, int(q))]]


    seen_alldiff_constraints = set()

    #Helper function to add unique AllDifferent constraints to the solver    
    def alldiff(vs):
        kk = tuple(sorted(id(x) for x in vs))
        if kk not in seen_alldiff_constraints:
            seen_alldiff_constraints.add(kk); md.AddAllDifferent(vs)

    # --- C1: ticks on each check are pairwise distinct ---
    for tk in list(tX.values()) + list(tZ.values()): alldiff(list(tk.values()))

    # --- C2: ticks on each data qubit are pairwise distinct ---
    perq = {}
    for tk in list(tX.values()) + list(tZ.values()):
        for q, v in tk.items(): perq.setdefault(q, []).append(v)
    for q, vs in perq.items(): alldiff(vs)

    # --- C3: for every overlapping (X-check, Z-check) pair with shared
    # support S, the number of qubits q in S where the X-tick precedes the
    # Z-tick must be even ---
    seen_parity_constraints = set()
    for tkx in tX.values():
        for tkz in tZ.values():
            S = set(tkx) & set(tkz)
            if not S: continue
            kk = tuple(sorted((id(tkx[c]), id(tkz[c])) for c in S))
            if kk in seen_parity_constraints: continue
            seen_parity_constraints.add(kk)
            bs = []
            for c in S:
                bb = md.NewBoolVar("")
                md.Add(tkx[c] < tkz[c]).OnlyEnforceIf(bb)
                md.Add(tkx[c] > tkz[c]).OnlyEnforceIf(bb.Not())
                #tkx[c] == tkz[c] is implicitly excluded by C1/C2
                bs.append(bb)
            h = md.NewIntVar(0, len(S), "")
            md.Add(sum(bs) == 2 * h)  # standard CP-SAT for sum(bs) is even

    # exclude is a list of previously found tau vectors
    for prev in (exclude or []):
        diff = []
        for o in range(norb):
            b = md.NewBoolVar(""); md.Add(V[o] != int(prev[o])).OnlyEnforceIf(b); diff.append(b)
        md.AddBoolOr(diff)

    sv = cp_model.CpSolver()
    sv.parameters.max_time_in_seconds = tlim
    #Modify below for a custom number of workers
    sv.parameters.num_workers = 8

    if seed is not None: sv.parameters.random_seed = int(seed)
    t0 = time.perf_counter()
    st = sv.Solve(md)
    dt = time.perf_counter() - t0
    vals = [sv.Value(v) for v in V] if st in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None
    #Return the results of the CP-SAT model
    return sv.StatusName(st), dt, vals

def lift_validate(Hx, Hz, orb, vals):
    """
    Step 2 of Local-ASC: lift a solved orbit assignment to a full schedule on every
    original edge of E(T). From PCMs Hx, Hz, orbit orb used in solve_equivariant and vals the output of the CP-SAT solver

    This function independently re-checks C1--C3 on the lifted schedule.

    Returns (TX, TZ, conf, bad, pairs, dt, ne):
      TX, TZ: the lifted per-check tick dicts, {row: {qubit: tick}}.
      conf: count of C1/C2 violations found = 0 if and only if the lifted schedule is fully valid on both.
      bad: count of overlapping (X,Z) check pairs whose C3 parity condition failed.
      pairs: total number of overlapping (X,Z) check pairs examined.
      dt: time for the lift step.
      ne: total number of edges in the lifted schedule (|E(T)|.
    """
    #Initialize time, dictionaries of the full graph
    t0 = time.perf_counter()
    TX = {}; TZ = {}
    for r in range(Hx.shape[0]):
        for q in np.nonzero(Hx[r])[0]: TX.setdefault(r, {})[int(q)] = vals[orb[("X", r, int(q))]]
    for r in range(Hz.shape[0]):
        for q in np.nonzero(Hz[r])[0]: TZ.setdefault(r, {})[int(q)] = vals[orb[("Z", r, int(q))]]

    # lift-only timing
    dt = time.perf_counter() - t0  

    #Number of edges in the full tanner graph
    ne = sum(len(v) for v in TX.values()) + sum(len(v) for v in TZ.values())

    # independent C1 check
    conf = sum(1 for tk in list(TX.values()) + list(TZ.values()) if len(set(tk.values())) != len(tk))

    # independent C2 check
    perq = {}
    for tk in list(TX.values()) + list(TZ.values()):
        for q, t in tk.items(): perq.setdefault(q, []).append(t)
    conf += sum(1 for tt in perq.values() if len(set(tt)) != len(tt))

    # independent C3 check
    bad = pairs = 0
    for tkx in TX.values():
        for tkz in TZ.values():
            S = set(tkx) & set(tkz)
            if not S: continue
            pairs += 1
            if sum(1 for q in S if tkx[q] < tkz[q]) % 2: bad += 1

    return TX, TZ, conf, bad, pairs, dt, ne

def generate_schedule(Hx, Hz, D, output_csv, solver_tlim=120, odd_order_automorphisms=True):
    """
    Generates a CNOT schedule using local_asc_generic with a timeout.
    """
    autos = automorphisms(Hx, Hz)
    odd_orders = sorted({order_of(a) for a in autos if order_of(a) > 1 and order_of(a) % 2 == 1})
    gens = [[a for a in autos if order_of(a) == odd_orders[-1]][0]] if odd_order_automorphisms else \
           ([a for a in autos if order_of(a) > 1][:1] if len(autos) > 1 else [])

    edges = [("X", r, int(q)) for r in range(Hx.shape[0]) for q in np.nonzero(Hx[r])[0]] + \
            [("Z", r, int(q)) for r in range(Hz.shape[0]) for q in np.nonzero(Hz[r])[0]]
    orb, norb = orbit_map(edges, gens)

    status, _, vals = solve_equivariant(Hx, Hz, orb, norb, int(D), tlim=solver_tlim)
    
    if vals is not None:
        TX, TZ, conf, bad, _, _, _ = lift_validate(Hx, Hz, orb, vals)
        if conf == 0 and bad == 0:
            with open(output_csv, "w") as f:
                f.write(f"# Local-ASC depth-{D} schedule; status: {status}\n")
                f.write("type,check,data_qubit,tick\n")
                for r in sorted(TX):
                    for q in sorted(TX[r]): f.write(f"X,{r},{q},{TX[r][q]}\n")
                for r in sorted(TZ):
                    for q in sorted(TZ[r]): f.write(f"Z,{r},{q},{TZ[r][q]}\n")
            return True, status
    return False, status

def ladder(Hx, Hz, edges, autos, D, skip, above, tlim_step):
    """
    Algorithm 1 over every enumerated subgroup, largest first; returns (gens, orb, norb, status, vals, tried).
    `skip`: a subgroup key (subgroup_key) already tried, e.g. the maxodd subgroup in the hybrid policy.
    `above`: only subgroups with |H| > above (hybrid, after the maxodd attempt merely ran out of time).
    """
    subs = sorted(((k, v) for k, v in enumerate_subgroups(autos).items() if k != skip and v[1] > above),
                  key=lambda kv: -kv[1][1])
    gens, orb, norb, status = [], None, len(edges), "NONE"
    tried = []
    for _, (gens, order) in subs:
        orb, norb = orbit_map(edges, gens)
        if not is_locally_injective(edges, orb)[0]:
            tried.append((order, norb, "VACUOUS", 0.0)); continue
        status, dt, vals = solve_equivariant(Hx, Hz, orb, norb, D, tlim_step)
        tried.append((order, norb, status, round(dt, 3)))
        if status == "OPTIMAL":
            return gens, orb, norb, status, vals, tried
    return gens, orb, norb, status, None, tried

def hybrid(Hx, Hz, edges, autos, D, tlim_step, tlim_hybrid, tlim_max):
    """
    maxodd first (HYBRID_CAP limit), then:
      - maxodd subgroup trivial, not well-formed, or INFEASIBLE -> ladder over every other subgroup, largest first
        (trivial group last);
      - maxodd merely out of time (UNKNOWN) -> ladder over the LARGER subgroups only (fewer variables, faster);
        if none solves, re-solve the maxodd subgroup from scratch with the TLIM limit (1800 s). A smaller subgroup is a larger, slower model,
        so it is not tried just because maxodd was slow (Tan2TNCb_666_4_65: C37 occasionally needs > 10 s, and
        the only smaller subgroup is the unreduced model).
        """
    gens = pick(autos, "maxodd")
    tried = []
    if gens:
        orb, norb = orbit_map(edges, gens)
        if is_locally_injective(edges, orb)[0]:
            status, dt, vals = solve_equivariant(Hx, Hz, orb, norb, D, tlim_hybrid)
            tried.append(("maxodd", order_of(gens[0]), norb, status, round(dt, 3)))
            if status == "OPTIMAL":
                return gens, orb, norb, status, vals, tried
        else:
            status = "VACUOUS"
            tried.append(("maxodd", order_of(gens[0]), norb, status, 0.0))
        Hmax = _expand_closure(gens)
        skip = subgroup_key(Hmax)
        
        if status == "UNKNOWN":
            g, o, n, st2, v, tr = ladder(Hx, Hz, edges, autos, D, skip=skip, above=len(Hmax), tlim_step=tlim_step)
            tried += [("ladder",) + t for t in tr]
            if st2 == "OPTIMAL":
                return g, o, n, st2, v, tried
            status, dt, vals = solve_equivariant(Hx, Hz, orb, norb, D, tlim_max)
            tried.append(("maxodd-resume", order_of(gens[0]), norb, status, round(dt, 3)))
            return gens, orb, norb, status, vals, tried
    else:
        tried.append(("maxodd", 1, len(edges), "trivial, skipped", 0.0))
        skip = None
    g, orb, norb, status, vals, tr = ladder(Hx, Hz, edges, autos, D, skip=skip, tlim_step=tlim_step, above=0)
    return g, orb, norb, status, vals, tried + [("ladder",) + t for t in tr]