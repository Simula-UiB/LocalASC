import numpy as np
import stim


# ################### #
# Circuit constructor #
# ################### #

def build_CSS_memory_circuit(Hx, Hz, schedule_rows, basis, rounds):
    """
    Builds a memory experiment circuit for 'X' or 'Z' -basis using one ancilla per check. The circuit builds using a CNOT schedule given by the schedule_rows.
    Hx and Hz are the PCMs of the CSS code. -> np.ndarray
    Schedule_rows is a list obtained from the load_schedule function- -> list[Tuple]
    rounds indicates how many rounds of syndrome extraction should be considered. -> int
    """

    #Extract the dimensions of PCMs to know the number of data qubits and the number of 'X', 'Z' checks (ancilla qubits) 
    mx, n = Hx.shape
    mz = Hz.shape[0]
    
    
    D_cx = max(t for *_, t in schedule_rows) + 1 #Circuit depth
 
    tx = -np.ones((mx, n), int)
    tz = -np.ones((mz, n), int)

    for typ, chk, dq, t in schedule_rows:
        (tx if typ == "X" else tz)[chk, dq] = t #Store the depth at which check m and data qubit n interact. If the value at tx[chk,dq] or tz[chk,dq] = -1, then the particular dq doesn't interact with that given X or Z check.

    #Verify that there is no double occupancy at any tick
    for j in range(n):
        used = sorted(list(tx[tx[:, j] >= 0, j]) + list(tz[tz[:, j] >= 0, j]))
        assert len(used) == len(set(used)), f"data qubit {j} used twice in one round -- invalid schedule"

    #Verify the given rank matches the computed rank
    rx, _, _ = rank2_rref(Hx)
    rz, _, _ = rank2_rref(Hz)
    k_expected = n - rx - rz

    #Compute logicals
    LZ = logicals_CSS(Hx, Hz, basis, n, k_expected)

    #Indexing of data qubits and X or Z -check ancilla qubits
    data = list(range(n))
    xa = [n + i for i in range(mx)]
    za = [n + mx + i for i in range(mz)]
    
    #Dictionary to store the schedule per layer
    layers = {}
    for typ, chk, dq, t in schedule_rows:
        layers.setdefault(t, []).append((typ, chk, dq))


    #Initialize an empty circuit
    c = stim.Circuit()

    #Initialize data qubits depending on the experiment type.
    if basis == "Z":
        c.append("R", data)
    else:
        c.append("RX", data)

    #Initialize ancillas
    c.append("RX", xa)
    c.append("R", za)

    #End initialization
    c.append("TICK")  # type: ignore

    #Initialize list for rec measurement results (To declare DETECTORS and OBSERVABLES later on)
    mrec = []

    #Loop over each syndrome extraction round
    for r in range(rounds):
        #Loop over each depth of the schedule
        for t in range(D_cx):
            cxs = []
            #Loop over the CNOT schedule for each schedule
            for typ, chk, dq in layers.get(t, []):
                #If X-check then the ancilla qubit is the control and the data qubit is the target 
                if typ == "X":
                    cxs += [n + chk, dq]
                else:
                    cxs += [dq, n + mx + chk]

            #Appends "CX"s in the required orders only if the list is not empty
            if cxs:
                c.append("CX", cxs)

            #End current depth
            c.append("TICK") # type: ignore


        #After t rounds measure ancillas and register their measurement 
        c.append("MRX", xa)
        for i in range(mx):
            mrec.append(("X", r, i)) #X-basis measurement of ancilla i at round r
        c.append("MR", za)
        for i in range(mz):
            mrec.append(("Z", r, i)) #Z-basis measurement of ancilla i at round r
        
        #End measurement round
        c.append("TICK") # type: ignore

    #Final data readout
    c.append("M" if basis == "Z" else "MX", data)
    for j in range(n):
        mrec.append(("D", rounds, j)) #Data i measurement at final round 

    #Add detectors to the circuit using the measurement records
    append_detectors(c, mrec, Hx, Hz, basis, rounds, LZ, mx, mz, n)

    return c

def append_detectors(c, mrec, Hx, Hz, basis, rounds, LZ, mx, mz, n):
    """
    Appends DETECTOR/OBSERVABLE_INCLUDE instructions, given
    a fully-built circuit plus the list of measurement-record keys
    in the exact order they were appended to the circuit: 'mrec'. 
    Using the tuple convention ("X", round, check_index), ("Z", round, check_index), ("D", rounds, data_qubit)for the final data-qubit measurement.
    Out of build_CSS_memory_circuit so it can be compatible with any circuit builder that wants to follow the same detecvtor logic and produces a similar mrec list.
    """
    #Get the total number of measurements performed
    tot = len(mrec)

    #Index each measurement with the tuple corresponding to the key
    pos = {kk: i for i, kk in enumerate(mrec)}

    def rec(kk):
        return stim.target_rec(pos[kk] - tot)

    #Determine detectors depending on the basis of the experiment.
    B1 = "Z" if basis == "Z" else "X"
    B2 = "X" if basis == "Z" else "Z"
    Hb1 = Hz if basis == "Z" else Hx
    mb1 = mz if basis == "Z" else mx

    # DETECTOR coordinate convention: [spatial_index, round, type]
    #   type 0 = X-check parity, type 1 = Z-check parity
    #   spatial_index for type-1 (Z) detectors is offset by mx so X/Z share one coordinate space without collision.
    
    # Add detectors for the first round of syndrome extraction, in this round we only put DETECTORS on the same basis as the experiment, otherwise we could get non-deterministic detector errors.
    for i in range(mb1):
        c.append("DETECTOR", [rec((B1, 0, i))], [i if B1 == "X" else mx + i, 0, 0 if B1 == "X" else 1]) 

    #DETECTORS for in-between rounds, now we also add second type of detectors.
    for r in range(1, rounds):
        for i in range(mb1):
            c.append("DETECTOR", [rec((B1, r, i)), rec((B1, r - 1, i))],
                      [i if B1 == "X" else mx + i, r, 0 if B1 == "X" else 1])
        mb2 = mx if B2 == "X" else mz
        for i in range(mb2):
            c.append("DETECTOR", [rec((B2, r, i)), rec((B2, r - 1, i))],
                      [i if B2 == "X" else mx + i, r, 0 if B2 == "X" else 1])
    
    #Last round detectors, targets are both the DATA readout and final round measurements for basis of experiment.
    for i in range(mb1):
        tg = [rec(("D", rounds, j)) for j in range(n) if Hb1[i, j]] + [rec((B1, rounds - 1, i))]
        c.append("DETECTOR", tg, [i if B1 == "X" else mx + i, rounds, 0 if B1 == "X" else 1])
    
    #Add observables using the logicals of the corresponding basis
    for li, v in enumerate(LZ):
        tg = [rec(("D", rounds, j)) for j in range(n) if v[j]]
        c.append("OBSERVABLE_INCLUDE", tg, li)


# ############# #
# CSS functions #
# ############# #

def logicals_CSS(Hx, Hz, basis, n, k_expected):
    """
    Standard rank/kernel logical-operator computation for CSS codes.
    """
    Ha, Hb = (Hx, Hz) if basis == "Z" else (Hz, Hx)
    _, Bb, _ = rank2_rref(Hb)
    _, Ra, pivx = rank2_rref(Ha)
    free = [c for c in range(n) if c not in pivx]
    ker = []
    for c in free:
        v = np.zeros(n, np.uint8)
        v[c] = 1
        for i, pc in enumerate(pivx):
            if Ra[i, c]:
                v[pc] ^= 1
        ker.append(v)
    red, lead, out = [], [], []

    def add(v, keep):
        v = v.copy()
        for r_, l_ in zip(red, lead):
            if v[l_]:
                v ^= r_
        if v.any():
            if keep:
                out.append(v.copy())
            red.append(v)
            lead.append(int(np.argmax(v)))

    for r_ in Bb:
        add(r_, False)
    for v in ker:
        add(v, True)
    assert len(out) == k_expected, f"expected {k_expected} logicals, got {len(out)}"
    return out


# ################### #
# Auxiliary functions #
# ################### #
def load_schedule(path):
    """
    Takes as input the path to a .csv file obtained by LocalASC and outputs a list of Tuples (type, check, data_qubit, tick).
    The tuples contain all information regarding how the circuit should behave.
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("type,"):
                continue
            typ, chk, dq, tick = line.split(",")
            rows.append((typ, int(chk), int(dq), int(tick)))
    return rows

def rank2_rref(M):
    """
    Computes the reduced row-echelon form (rref) of a matrix.
    Outputs: rank, M in rref, and the list of pivots.
    """
    M = (M.copy() % 2).astype(np.uint8)
    r, piv = 0, []
    for c in range(M.shape[1]):
        w = np.where(M[r:, c] == 1)[0]
        if len(w) == 0:
            continue
        p = w[0] + r
        M[[r, p]] = M[[p, r]]
        for i in range(M.shape[0]):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        piv.append(c)
        r += 1
        if r == M.shape[0]:
            break
    return r, M[:r], piv