import time
import json
import multiprocessing
from pathlib import Path
import stim
import sinter
from stimbposd import SinterDecoder_BPOSD

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_SINTER_DIR = PROJECT_ROOT / "results" / "sinter_data"
from src.LocalASC.circuit_builder.noise import add_circuit_level_noise



#############################
# Sinter decoder experiments #
#############################

def evaluate_circuit_fast(base_circ_x, base_circ_z, p, d, num_shots, max_errors, max_bp_iter):
    """
    Runs a quick Sinter memory experiment to score the schedule for circuit filetring.
    """
    tasks = []
    noisy_x = add_circuit_level_noise(base_circ_x, p1q=p, p2q=p, p_meas=p)
    noisy_z = add_circuit_level_noise(base_circ_z, p1q=p, p2q=p, p_meas=p)
    
    tasks.append(sinter.Task(circuit=noisy_x, json_metadata={'basis': 'X'}))
    tasks.append(sinter.Task(circuit=noisy_z, json_metadata={'basis': 'Z'}))

    custom_decoders = {
        'bposd_cs': SinterDecoder_BPOSD(
            max_bp_iters=max_bp_iter, 
            bp_method="ms", 
            osd_method="osd_cs", 
            osd_order=10
        )
    }

    stats_list = list(sinter.collect(
        num_workers=max(1, multiprocessing.cpu_count() // 3),
        tasks=tasks,
        decoders=['bposd_cs'],
        custom_decoders=custom_decoders, #type: ignore
        max_shots=num_shots,
        max_errors=max_errors,
        print_progress=False  # Keep console clean during massive loops
    ))

    results = {stat.json_metadata['basis']: stat for stat in stats_list}

    nx_e, nx_s = results['X'].errors, results['X'].shots
    nz_e, nz_s = results['Z'].errors, results['Z'].shots

    raw_x = nx_e / nx_s if nx_s > 0 else 1.0
    raw_z = nz_e / nz_s if nz_s > 0 else 1.0
    
    pl_x = 1.0 - (1.0 - raw_x) ** (1.0 / d)
    pl_z = 1.0 - (1.0 - raw_z) ** (1.0 / d)
    
    return 1.0 - (1.0 - pl_z) * (1.0 - pl_x), nx_e, nz_e

def build_sinter_task_memory_experiment(circuit : stim.Circuit, physical_error_probs : list[float], basis : str) -> list[sinter.Task]:
    tasks = []
    for p in physical_error_probs:
        noisy_circuit = add_circuit_level_noise(circuit, p1q=p, p2q=p, p_meas=p)
        tasks.append(sinter.Task(circuit=noisy_circuit, json_metadata={'p' : p, 'basis': basis}))
    return tasks

def run_sinter_experiment(
    base_circ_x: stim.Circuit, 
    base_circ_z: stim.Circuit, 
    physical_error_probs: list[float], 
    filename: str, 
    name: str, 
    n: int, 
    k: int, 
    nrounds: int,
    num_shots: int, 
    max_errors: int, 
    max_bp_iter: int,
    d: int | None = None,
    output_dir: str | Path = DEFAULT_SINTER_DIR,
    verbose= False
):
    """
    Runs decoding for given X and Z memory experiment circuits.
    """

    #Destination path
    safe_filename = Path(filename).name
    filepath = Path(output_dir) / safe_filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    #Add run id to distinguish different runs with the same parameters in the .json 
    run_id = int(time.time())

    print(f"Generating Sinter tasks for {len(physical_error_probs)} noise levels...")
    
    # Build all sinter tasks
    tasks = build_sinter_task_memory_experiment(base_circ_x, physical_error_probs, 'X')
    tasks = tasks + build_sinter_task_memory_experiment(base_circ_z, physical_error_probs, 'Z')

    # Can edit to add other decoders
    custom_decoders = {
        'bposd_cs': SinterDecoder_BPOSD(
            max_bp_iters=max_bp_iter,
            bp_method="ms",
            osd_method="osd_cs",
            osd_order=10
        ) 
    }
    
    start_time = time.perf_counter()
    
    stats_list = list(sinter.collect(
        num_workers=max(1, multiprocessing.cpu_count() // 2), #modify this for the number of workers you want to use 
        tasks=tasks,
        decoders=['bposd_cs'],
        custom_decoders=custom_decoders,  #type: ignore
        max_shots=num_shots,
        max_errors=max_errors,
        print_progress= verbose # Toggle verbose if you want to print progress
    ))
    
    elapsed_time = time.perf_counter() - start_time
    print(f"\nSinter execution complete in {elapsed_time:.2f}s.")

    stats_map = {p: {} for p in physical_error_probs}
    for stat in stats_list:
        p_val = stat.json_metadata['p']
        basis_val = stat.json_metadata['basis']
        stats_map[p_val][basis_val] = stat

    records_to_save = []
    
    for p in physical_error_probs:
        stat_x = stats_map[p]['X']
        stat_z = stats_map[p]['Z']
        
        n_e_x, n_s_x = stat_x.errors, stat_x.shots
        n_e_z, n_s_z = stat_z.errors, stat_z.shots
        
        print(f"\nResults for p = {p:.4e}:")
        print(f"  > X-Basis: {n_e_x}/{n_s_x} mistakes")
        print(f"  > Z-Basis: {n_e_z}/{n_s_z} mistakes")
        
        record = {
            "name": name,
            "decoder": "sinter_bposd",
            "n": n,
            "k": k,
            "d": d,
            "rounds": nrounds,
            "p": p,
            "shots_x": int(n_s_x),
            "errors_x": int(n_e_x),
            "shots_z": int(n_s_z),
            "errors_z": int(n_e_z),
            "max_bp_iter": max_bp_iter,
            "batch_compute_time": round(elapsed_time, 2),
            "run_id": run_id
        }
        
        records_to_save.append(json.dumps(record))

    #Write results to file
    if records_to_save:
        with open(filepath, 'a') as f: 
            f.write('\n'.join(records_to_save) + '\n')
            
    print(f"Wrote Sinter results to {filepath}")