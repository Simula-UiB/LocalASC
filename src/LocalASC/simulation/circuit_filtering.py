import time
import json
import shutil
from pathlib import Path
import numpy as np

from circuit_builder.circuit_builder import build_CSS_memory_circuit
from sinter_experiment import evaluate_circuit_fast
from scheduling.local_ASC_scheduler import generate_schedule
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_BEST_DIR = PROJECT_ROOT / "results" / "best_circuits"
DEFAULT_TEMP_DIR = PROJECT_ROOT / "results" / "optimization_runs"

#############
# Filtering #
#############

def run_filtering_pipeline(
    Hx: np.ndarray, 
    Hz: np.ndarray, 
    rounds: int, 
    max_time_hrs: float, 
    eval_p: float, 
    eval_shots: int, 
    eval_errors: int,
    target_depth: int,
    number_of_searched_circs: int
):
    """
    Master loop that generates, builds, tests, and saves the best circuit.
    """

    target_depths = np.ones(number_of_searched_circs)*target_depth # Modify this if want to look for diferent depths
    
    run_id = int(time.time())
    out_dir = DEFAULT_TEMP_DIR / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    DEFAULT_BEST_DIR.mkdir(parents=True, exist_ok=True)
    
    global_start_time = time.time()
    max_time_sec = max_time_hrs * 3600

    best_ler = float('inf')
    best_config = None
    curr_step = 0

    print(f"=== Starting Circuit Filtering Pipeline ===")
    print(f"Temporary Run Directory: {out_dir}")
    print(f"Filter evaluating at p={eval_p}")

    for D in target_depths:
        curr_step += 1
        if (time.time() - global_start_time) > max_time_sec:
            print("\n[!] Global timeout reached. Terminating search.")
            break

        print(f"\n--- Exploring Depth D={D} (Step {curr_step}) ---")
        step_prefix = out_dir / f"step_{curr_step}"
        schedule_csv = f"{step_prefix}_schedule.csv"
        
        # Generate schedules
        success, status = generate_schedule(Hx, Hz, int(D), schedule_csv, solver_tlim=120)
        if not success:
            print(f"Schedule generation failed or timed out (Status: {status}) try a new target depth or run again.")
            break

        # Build
        circ_x = build_CSS_memory_circuit(Hx, Hz, schedule_csv, "X", rounds)
        circ_z = build_CSS_memory_circuit(Hx, Hz, schedule_csv, "Z", rounds)
        
        circ_x.to_file(f"{step_prefix}_memX.stim")
        circ_z.to_file(f"{step_prefix}_memZ.stim")

        # Evaluate
        ler, err_x, err_z = evaluate_circuit_fast(circ_x, circ_z, eval_p, rounds, eval_shots, eval_errors, max_bp_iter=10)

        print(f"Evaluated LER: {ler:.3e} (X errs: {err_x}, Z errs: {err_z})")

        # Compare & Save Winner
        step_result = {
            "status": status,
            "depth": int(D),
            "eval_p": eval_p,
            "total_ler": ler,
            "errors_x": err_x,
            "errors_z": err_z,
            "timestamp": time.time() - global_start_time
        }

        with open(f"{step_prefix}_results.json", "w") as f:
            json.dump(step_result, f, indent=4)

        if ler < best_ler:
            best_ler = ler
            best_config = step_result
            print(f"*** New best circuit found! LER: {best_ler:.3e} ***")
            
            # Copy winners to the permanent best_circuits folder
            shutil.copy(schedule_csv, DEFAULT_BEST_DIR / "BEST_schedule.csv")
            shutil.copy(f"{step_prefix}_memX.stim", DEFAULT_BEST_DIR / "BEST_memX.stim")
            shutil.copy(f"{step_prefix}_memZ.stim", DEFAULT_BEST_DIR / "BEST_memZ.stim")
            with open(DEFAULT_BEST_DIR / "BEST_metadata.json", "w") as f:
                json.dump(best_config, f, indent=4)

    print("\n=== Filtering Complete ===")
    if best_config:
        print(f"Best configuration: Depth {best_config['depth']} with LER {best_config['total_ler']:.3e}")
        print(f"Saved securely in {DEFAULT_BEST_DIR}")
    else:
        print("No valid circuits were generated.")
