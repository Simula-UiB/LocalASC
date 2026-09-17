import os
import sys
import multiprocessing

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import json
from ldpc.bposd_decoder import BpOsdDecoder
import time
from circuit_builder.apply_noise import add_circuit_level_noise, get_noisy_circuit
import argparse
import stim
from ldpc.sinter_decoders import sinter_bposd_decoder
from stimbposd import SinterDecoder_BPOSD
import matplotlib.pyplot as plt
from stimbposd import BPOSD
from tqdm import tqdm
import sinter

worker_env = {}

def init_worker(circuit_str, max_iter):
    """
    Runs exactly ONCE per CPU core when the pool starts. 
    It builds the heavy BPOSD matrices and keeps them in memory.
    """
    noisy_circ = stim.Circuit(circuit_str)
    dem = noisy_circ.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True)
    
    worker_env['decoder'] = BPOSD(
        dem, 
        max_bp_iters=max_iter,       
        bp_method="ms",           
        osd_order=10,             
        osd_method="osd_cs"
    )
    worker_env['circuit'] = noisy_circ

def decode_worker(shots_to_run, worker_seed):
    """
    Runs multiple times. Reuses the already-built decoder for maximum speed.
    """
    # Compiling the sampler locally is extremely fast
    sampler = worker_env['circuit'].compile_detector_sampler(seed=worker_seed)
    shots, observables = sampler.sample(shots_to_run, separate_observables=True, bit_packed=False)
    
    # Instantly decode using the pre-initialized decoder in memory
    predicted_observables = worker_env['decoder'].decode_batch(shots)
    
    return int(np.sum(np.any(predicted_observables != observables, axis=1)))

def simulate_single_basis(
    base_circ: stim.Circuit, 
    p: float, 
    num_shots: int, 
    max_errors: int, 
    max_iter: int, 
    seed: int,
    basis_name: str
):
    noisy_circ, idle_count = add_circuit_level_noise(base_circ, p1q=p, p2q=p, p_meas=p)
    
    num_mistakes = 0
    shots_simulated = 0
    
    cores = multiprocessing.cpu_count() // 2
    chunk_per_core = max(1000, 10000 // cores) 
    
    # Convert circuit to string so it can safely pass into the worker initialization
    circ_str = str(noisy_circ)
    
    # Initialize the pool WITH the init_worker function
    with multiprocessing.Pool(
        processes=cores, 
        initializer=init_worker, 
        initargs=(circ_str, max_iter)
    ) as pool, tqdm(total=num_shots, desc=f"  {basis_name}-Basis", leave=False) as pbar:
        
        while shots_simulated < num_shots and num_mistakes < max_errors:
            remaining_shots = num_shots - shots_simulated
            
            jobs = []
            shots_this_round = 0
            
            for i in range(cores):
                if shots_this_round >= remaining_shots:
                    break
                s = min(chunk_per_core, remaining_shots - shots_this_round)
                
                worker_seed = seed + shots_simulated + shots_this_round + i 
                
                # Note: We no longer send the noisy_circ in the job args!
                jobs.append((s, worker_seed))
                shots_this_round += s
            
            # Execute jobs
            results = pool.starmap(decode_worker, jobs)
            
            chunk_mistakes = sum(results)
            num_mistakes += chunk_mistakes
            shots_simulated += shots_this_round
            
            pbar.set_postfix(errors=f"{num_mistakes}/{max_errors}")
            pbar.update(shots_this_round)
            
    return num_mistakes, shots_simulated, idle_count

def run_batch_bposd_experiment(
    base_circ_x: stim.Circuit,
    base_circ_z: stim.Circuit,
    phys_err_probs: list[float],
    output_filepath: str,
    name: str,
    n: int,
    k: int,
    d: int, 
    num_shots: int,
    max_errors: int,
    max_iter: int,
    seed: int
):
    for p in phys_err_probs:
        print(f"\nEvaluating p = {p:.4e}...")
        start_time = time.perf_counter()
        
        # Simulate bases independently
        n_e_x, n_s_x, idle_x = simulate_single_basis(base_circ_x, p, num_shots, max_errors, max_iter, seed, 'X')
        n_e_z, n_s_z, idle_z = simulate_single_basis(base_circ_z, p, num_shots, max_errors, max_iter, seed, 'Z')
        
        # Logical Error Rates (Block / raw rates)
        raw_x = n_e_x / n_s_x if n_s_x > 0 else 0.0
        raw_z = n_e_z / n_s_z if n_s_z > 0 else 0.0
        
        # Per-Round LER using formula: 1 - (1 - N_e / N_s)^(1/N_rounds)
        pl_x = 1.0 - (1.0 - raw_x) ** (1.0 / d)
        pl_z = 1.0 - (1.0 - raw_z) ** (1.0 / d)
        
        #Total Aggregated LER
        ler_total = 1.0 - (1.0 - pl_z) * (1.0 - pl_x)
                
        elapsed_time = time.perf_counter() - start_time
        
        print(f"  > X-Basis: {n_e_x}/{n_s_x} mistakes (pl,x = {pl_x:.4g})")
        print(f"  > Z-Basis: {n_e_z}/{n_s_z} mistakes (pl,z = {pl_z:.4g})")
        print(f"  > Aggregated Total LER : {ler_total:.4g}")
        print(f"  > Compute Time         : {elapsed_time:.2f}s")
        
        # 5. Format and save metadata to JSON
        record = {
            "name": name,
            "decoder": "bposd_batch",
            "n": n,
            "k": k,
            "d": d,
            "rounds": d,
            "p": p,
            "shots_x": int(n_s_x),
            "errors_x": int(n_e_x),
            "shots_z": int(n_s_z),
            "errors_z": int(n_e_z),
            "ler_total": float(ler_total),
            "seed": seed,
            "max_iter": max_iter,
            "decode_s": round(elapsed_time, 2)
        }
        
        with open(output_filepath, 'a') as f: 
            f.write(json.dumps(record) + '\n')
            
    print(f"\nCompleted total LER computation. Base idle errors: X={idle_x}, Z={idle_z}")

def run_sinter_experiment(
    base_circ_x: stim.Circuit,
    base_circ_z: stim.Circuit,
    phys_err_probs: list[float],
    output_filepath: str,
    name: str,
    n: int,
    k: int,
    d: int, 
    num_shots: int,
    max_errors: int,
    max_iter: int,
    seed: int
):
    tasks = []
    idle_counts = {}

    print(f"Generating Sinter tasks for {len(phys_err_probs)} noise levels...")
    
    # 1. Build all tasks up front
    for p in phys_err_probs:
        noisy_x, idle_x = add_circuit_level_noise(base_circ_x, p1q=p, p2q=p, p_meas=p)
        noisy_z, idle_z = add_circuit_level_noise(base_circ_z, p1q=p, p2q=p, p_meas=p)
        
        idle_counts[(p, "X")] = idle_x
        idle_counts[(p, "Z")] = idle_z
        
        tasks.append(sinter.Task(
            circuit=noisy_x,
            json_metadata={'p': p, 'basis': 'X'}
        ))
        tasks.append(sinter.Task(
            circuit=noisy_z,
            json_metadata={'p': p, 'basis': 'Z'}
        ))

   
    custom_decoders = {
        'bposd_cs': SinterDecoder_BPOSD(
            max_bp_iters=max_iter,
            bp_method="ms",
            osd_method="osd_cs",
            osd_order=10
        )
    }

    
    print(f"Executing {len(tasks)} tasks across {3 * multiprocessing.cpu_count() // 4} cores...")
    start_time = time.perf_counter()
    
    stats_list = sinter.collect(
        num_workers= 3 * multiprocessing.cpu_count() // 4,
        tasks=tasks,
        decoders=['bposd_cs'],
        custom_decoders=custom_decoders, #type: ignore
        max_shots=num_shots,
        max_errors=max_errors,
        print_progress=True
    )
    
    elapsed_time = time.perf_counter() - start_time
    print(f"\nSinter execution complete in {elapsed_time:.2f}s.")

   
    stats_map = {p: {} for p in phys_err_probs}
    for stat in stats_list:
        p_val = stat.json_metadata['p']
        basis_val = stat.json_metadata['basis']
        stats_map[p_val][basis_val] = stat

    
    for p in phys_err_probs:
        stat_x = stats_map[p]['X']
        stat_z = stats_map[p]['Z']
        
        n_e_x, n_s_x = stat_x.errors, stat_x.shots
        n_e_z, n_s_z = stat_z.errors, stat_z.shots
        
        raw_x = n_e_x / n_s_x if n_s_x > 0 else 0.0
        raw_z = n_e_z / n_s_z if n_s_z > 0 else 0.0
        
        pl_x = 1.0 - (1.0 - raw_x) ** (1.0 / d)
        pl_z = 1.0 - (1.0 - raw_z) ** (1.0 / d)
        
        ler_total = 1.0 - (1.0 - pl_z) * (1.0 - pl_x)
        
        print(f"\nResults for p = {p:.4e}:")
        print(f"  > X-Basis: {n_e_x}/{n_s_x} mistakes (pl,x = {pl_x:.4g})")
        print(f"  > Z-Basis: {n_e_z}/{n_s_z} mistakes (pl,z = {pl_z:.4g})")
        print(f"  > Aggregated Total LER : {ler_total:.4g}")
        
        record = {
            "name": name,
            "decoder": "sinter_bposd",
            "n": n,
            "k": k,
            "d": d,
            "rounds": d,
            "p": p,
            "shots_x": int(n_s_x),
            "errors_x": int(n_e_x),
            "shots_z": int(n_s_z),
            "errors_z": int(n_e_z),
            "ler_total": float(ler_total),
            "idle_errors_x": idle_counts[(p, "X")],
            "idle_errors_z": idle_counts[(p, "Z")],
            "seed": seed,
            "max_iter": max_iter,
            "batch_compute_time": round(elapsed_time, 2)
        }
        
        with open(output_filepath, 'a') as f: 
            f.write(json.dumps(record) + '\n')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    phys_err_probs = [0.0002, 0.0004, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006]

    # Required parameters
    parser.add_argument("-n", type=int, required=True, help="Number of physical qubits (n)")
    parser.add_argument("-k", type=int, required=True, help="Number of logical qubits (k)")
    parser.add_argument("-d", type=int, required=True, help="Code distance (d)")
    parser.add_argument("-name", type=str, required=True, help="Select a type of circuit from ATB, localASC and fullASC")

    # Optional parameters
    parser.add_argument("-shots", type=int, required=False, default=1000, help="Number of shots")
    parser.add_argument("-max_iter", type=int, required=False, default=20, help="Maximum number of BP iterations")
    parser.add_argument("-seed", type=int, required=False, default=42, help="Random seed")
    parser.add_argument("-max_errors", type=int, required=False, default=50, help="Maximum number of errors collected before moving on")
    parser.add_argument("--free", action="store_true", help= "Choose one of four circuits free_depth7_rounds7_mem, depth7_rounds7_mem, free_depth7_mem or depth7_mem")
    parser.add_argument("--rounds", action="store_true", help= "Choose one of four circuits free_depth7_rounds7_mem, depth7_rounds7_mem, free_depth7_mem or depth7_mem")
    parser.add_argument("--nrounds", type=int, required=False, default= 6, help="Number of rounds")

    args = parser.parse_args()
    
    # Create the results directory and filepath
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Circuit Directions
    if args.n == 48:
        folder_name = project_root + f"/Codes/48_{args.k}_Coset2BGA/circuits/"
        depth = 7 if args.name == 'ATB_official' else 6
        file_direction = folder_name + f'48_{args.k}_Coset2BGA_circuit_{args.name}_rounds{args.nrounds}_depth{depth}_mem'
        custom_filepath = os.path.join(output_dir, f"{args.name}_n{args.n}_k{args.k}_d{args.d}_depth{depth}_nshots{args.shots}.json")
    else:
        folder_name = project_root + f"/Codes/BB_Codes/BB_{args.n}_{args.k}_{args.d}/circuits_ASC/"
        name_tag = 'ATB_official' if args.name == 'ATB' else args.name
        file_pref = folder_name + f'BB_{args.n}_{args.k}_{args.d}_circuit_{name_tag}_'
        if args.free and args.rounds:
            file_direction = file_pref + f'free_depth7_rounds7_mem'
            custom_filepath = os.path.join(output_dir, f"{args.name}_n{args.n}_k{args.k}_d{args.d}_free_depth7_rounds7_nshots{args.shots}.json")
        elif args.free and not args.rounds:
            file_direction = file_pref + f'free_depth7_mem'
            custom_filepath = os.path.join(output_dir, f"{args.name}_n{args.n}_k{args.k}_d{args.d}_free_depth7_nshots{args.shots}.json")
        elif not args.free and args.rounds:
            file_direction = file_pref + f'depth7_rounds7_mem'
            custom_filepath = os.path.join(output_dir, f"{args.name}_n{args.n}_k{args.k}_d{args.d}_depth7_rounds7_nshots{args.shots}.json")
        else:
            file_direction = file_pref + f'depth7_mem'
            custom_filepath = os.path.join(output_dir, f"{args.name}_n{args.n}_k{args.k}_d{args.d}_depth7_nshots{args.shots}.json")

    x_file_direction = file_direction + 'X.stim'
    z_file_direction = file_direction + 'Z.stim'

    base_circ_x = stim.Circuit.from_file(x_file_direction)
    base_circ_z = stim.Circuit.from_file(z_file_direction)

    print(f"Starting batch BP-OSD decoding for {args.name} at {file_direction} \n (Results -> {custom_filepath})")
    
    # Execute the experiment
    """ run_batch_bposd_experiment(
    base_circ_x= base_circ_x,
    base_circ_z= base_circ_z,
    phys_err_probs=phys_err_probs,
    output_filepath=custom_filepath,
    name=args.name,
    n=args.n,
    k= args.k,
    d= args.d+1,
    num_shots=args.shots,
    max_errors=args.max_errors,
    max_iter=args.max_iter,
    seed= args.seed
    ) """
    run_sinter_experiment(
    base_circ_x= base_circ_x,
    base_circ_z= base_circ_z,
    phys_err_probs=phys_err_probs,
    output_filepath=custom_filepath,
    name=args.name,
    n=args.n,
    k= args.k,
    d= args.d,
    num_shots=args.shots,
    max_errors=args.max_errors,
    max_iter=args.max_iter,
    seed= args.seed
    )