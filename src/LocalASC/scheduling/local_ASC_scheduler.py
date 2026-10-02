import networkx as nx
import numpy as np
from networkx.algorithms.isomorphism import GraphMatcher, categorical_node_match
import pynauty
############
# Orbit finding  #
############



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


def orbit_map(edges, gens):
    """
    
    """
    parent = {e: e for e in edges}

    def find(x):
        # path-compressed find: also rewrites parent[x] to its grandparent on
        # the way, which is what gives the "one pass is enough" property above
        # its practical efficiency (not just its correctness).
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb

    for g in gens:
        for e in edges: union(e, edge_action(g, e))

    # relabel each orbit's arbitrary union-find root with a dense 0..norb-1 id,
    # in first-seen order, so the ids can index directly into a CP-SAT variable
    # array (see V = [... for _ in range(norb)] in solve_equivariant()).
    reps = {}
    out = {}
    for e in edges:
        r = find(e)
        if r not in reps: reps[r] = len(reps)
        out[e] = reps[r]
    return out, len(reps)