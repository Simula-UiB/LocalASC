import os
import sys
import multiprocessing

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import json
from ldpc.bposd_decoder import BpOsdDecoder
from scipy.sparse import csc_matrix
import time
from circuit_builder.apply_noise import add_circuit_level_noise
import sinter
import argparse
import stim
import matplotlib.pyplot as plt
from stimbposd import BPOSD



def dem_to_matrices(dem):
    cols = {}
    for inst in dem.flattened():
        if inst.type != 'error': continue
        p = inst.args_copy()[0]
        dets = set(); obs = set()
        for t in inst.targets_copy():
            if t.is_relative_detector_id(): dets.add(t.val)
            elif t.is_logical_observable_id(): obs.add(t.val)
        key = (frozenset(dets), frozenset(obs))
        if key in cols: cols[key] = cols[key]*(1-p) + p*(1-cols[key])
        else: cols[key] = p
    nd, no = dem.num_detectors, dem.num_observables
    keys = list(cols)
    ri, ci, oi, oc, pr = [], [], [], [], []
    for j, (ds, os_) in enumerate(keys):
        for dd in ds: ri.append(dd); ci.append(j)
        for oo in os_: oi.append(oo); oc.append(j)
        pr.append(cols[(ds, os_)])
    H = csc_matrix((np.ones(len(ri), np.uint8), (ri, ci)), shape=(nd, len(keys)))
    O = csc_matrix((np.ones(len(oi), np.uint8), (oi, oc)), shape=(no, len(keys)))
    return H, O, np.array(pr)

def run_batch_bposd_experiment(
    base_circ: stim.Circuit,
    phys_err_probs: list[float],
    output_filepath: str,
    name: str,
    n: int,
    k: int,
    d: int,
    basis: str,
    num_shots: int,
    max_iter: int,
    seed: int
):
    for p in phys_err_probs:
        print(f"\nEvaluating p = {p:.4e}...")
        start_time = time.perf_counter()
        
        # 1. Apply noise
        noisy_circ, idle_count = add_circuit_level_noise(base_circ, p1q=p, p2q=p, p_meas=p)
        
        # 2. Sample the circuit for syndromes (shots) and actual observables
        sampler = noisy_circ.compile_detector_sampler(seed=seed)
        shots, observables = sampler.sample(num_shots, separate_observables=True, bit_packed=False)
        
        # 3. Initialize decoder directly from DEM and decode batch
        decoder = BPOSD(noisy_circ.detector_error_model(decompose_errors=True), max_bp_iters=max_iter)
        predicted_observables = decoder.decode_batch(shots)
        
        # 4. Vectorized error counting
        num_mistakes = np.sum(np.any(predicted_observables != observables, axis=1))
                
        elapsed_time = time.perf_counter() - start_time
        ler = num_mistakes / num_shots
        
        print(f"  > {num_mistakes}/{num_shots} mistakes (LER = {ler:.4g})")
        print(f"  > Compute Time   : {elapsed_time:.2f}s")
        
        # 5. Format and save metadata to JSON
        record = {
            "name": name,
            "decoder": "bposd_batch",
            "n": n,
            "k": k,
            "d": d,
            "rounds": d,
            "basis": basis,
            "p": p,
            "shots": num_shots,
            "errors": int(num_mistakes),
            "ler": float(ler),
            "seed": seed,
            "max_iter": max_iter,
            "decode_s": round(elapsed_time, 2)
        }
        
        with open(output_filepath, 'a') as f: 
            f.write(json.dumps(record) + '\n')
    print(f"Circuit with {idle_count} idle errors LER computed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    phys_err_probs = list(np.logspace(-4, -2, 7, dtype=float))

    # Required parameters
    parser.add_argument("-n", type=int, required=True, help="Number of physical qubits (n)")
    parser.add_argument("-k", type=int, required=True, help="Number of logical qubits (k)")
    parser.add_argument("-d", type=int, required=True, help="Code distance (d)")
    parser.add_argument("-name", type=str, required=True, help="Select a type of circuit from ATB, localASC and fullASC")

    # Optional parameters
    parser.add_argument("-shots", type=int, required=False, default=1000, help="Number of shots")
    parser.add_argument("-max_iter", type=int, required=False, default=20, help="Maximum number of BP iterations")
    parser.add_argument("-seed", type=int, required=False, default=42, help="Random seed")
    parser.add_argument("-basis", type=str, default='Z', help="Basis for memory experiment")

    parser.add_argument("--free", action="store_true", help= "Choose one of four circuits free_depth7_rounds7_mem, depth7_rounds7_mem, free_depth7_mem or depth7_mem")
    parser.add_argument("--rounds", action="store_true", help= "Choose one of four circuits free_depth7_rounds7_mem, depth7_rounds7_mem, free_depth7_mem or depth7_mem")

    args = parser.parse_args()
    
    # Create the results directory and filepath
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    custom_filepath = os.path.join(output_dir, f"{args.name}_n{args.n}_k{args.k}_d{args.d}_nshots{args.shots}.json")

    # Circuit Directions
    if args.n == 48:
        folder_name = "Codes/48_4_Coset2BGA/circuits/"
        depth = 7 if args.name == 'ATB_official' else 6
        file_direction = folder_name + f'48_4_Coset2BGA_circuit_{args.name}_depth{depth}_mem{args.basis}.stim'
    else:
        folder_name = f"Codes/BB_codes/BB_{args.n}_{args.k}_{args.d}/circuits_ASC/"
        name_tag = 'ATB_official' if args.name == 'ATB' else args.name
        file_pref = folder_name + f'BB_{args.n}_{args.k}_{args.d}_circuit_{name_tag}_'
        if args.free and args.rounds:
            file_direction = file_pref + f'free_depth7_rounds7_mem{args.basis}.stim'
        elif args.free and not args.rounds:
            file_direction = file_pref + f'free_depth7_mem{args.basis}.stim'
        elif not args.free and args.rounds:
            file_direction = file_pref + f'depth7_rounds7_mem{args.basis}.stim'
        else:
            file_direction = file_pref + f'depth7_mem{args.basis}.stim'

    base_circ = stim.Circuit.from_file(file_direction)
    
    print(f"Starting batch BP-OSD decoding for {args.name} at {file_direction} \n (Results -> {custom_filepath})")
    
    # Execute the experiment
    run_batch_bposd_experiment(
        base_circ=base_circ,
        phys_err_probs=phys_err_probs,
        output_filepath=custom_filepath,
        name=args.name,
        n=args.n,
        k=args.k,
        d=args.d,
        basis=args.basis,
        num_shots=args.shots,
        max_iter=args.max_iter,
        seed=args.seed
    )