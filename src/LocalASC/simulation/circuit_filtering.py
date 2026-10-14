import time
import json
import shutil
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_BEST_DIR = PROJECT_ROOT / "results" / "best_circuits"
DEFAULT_TEMP_DIR = PROJECT_ROOT / "results" / "optimization_runs"

from src.LocalASC.circuit_builder.circuit_builder import build_CSS_memory_circuit, rank2_rref
from src.LocalASC.simulation.sinter_experiment import evaluate_circuit_fast
from LocalASC.scheduling.localASC_scheduler import generate_schedule


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

    n_qubits = Hx.shape[1]
    kx, _, _ = rank2_rref(Hx)
    kz, _, _ = rank2_rref(Hz) 
    logical_qubits = n_qubits - kx - kz 
    
    target_depths = np.ones(number_of_searched_circs)*target_depth # Modify this if want to look for diferent depths
    
    run_id = int(time.time())
    out_dir = DEFAULT_TEMP_DIR / f"run_{run_id}"
    circuits_dir = out_dir / "circuits"
    csv_dir = out_dir / "schedule_csv"
    json_dir = out_dir / "schedule_json"
    
    for d in [circuits_dir, csv_dir, json_dir, DEFAULT_BEST_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    
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

        print(f"\n--- Exploring Depth D={int(D)} (Step {curr_step}) ---")
        
        schedule_csv = csv_dir / f"step_{curr_step}_schedule.csv"
        circ_x_path = circuits_dir / f"step_{curr_step}_memX.stim"
        circ_z_path = circuits_dir / f"step_{curr_step}_memZ.stim"
        result_json_path = json_dir / f"step_{curr_step}_results.json"
        
        # Generate schedules
        success, status = generate_schedule(Hx, Hz, int(D), schedule_csv, solver_tlim=120)
        if not success:
            print(f"Schedule generation failed or timed out (Status: {status}) try a new target depth or run again.")
            break

        # Build
        circ_x = build_CSS_memory_circuit(Hx, Hz, schedule_csv, "X", rounds)
        circ_z = build_CSS_memory_circuit(Hx, Hz, schedule_csv, "Z", rounds)
        
        circ_x.to_file(str(circ_x_path))
        circ_z.to_file(str(circ_z_path))

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

        with open(result_json_path, "w") as f:
            json.dump(step_result, f, indent=4)

        if ler < best_ler:
            best_ler = ler
            best_config = step_result
            print(f"*** New best circuit found! LER: {best_ler:.3e} ***")

            winner_dir = DEFAULT_BEST_DIR / f"code_n{n_qubits}_k{logical_qubits}_r{rounds}"
            for sub_dir in ["schedule_csv", "circuits", "schedule_json"]:
                (winner_dir / sub_dir).mkdir(parents=True, exist_ok=True)
            
            # Copy winners to the permanent best_circuits folder
            shutil.copy(schedule_csv, winner_dir / "schedule_csv/BEST_schedule.csv")
            shutil.copy(circ_x_path, winner_dir / "circuits/BEST_memX.stim")
            shutil.copy(circ_z_path, winner_dir / "circuits/BEST_memZ.stim")
            
            with open(winner_dir / "schedule_json/BEST_metadata.json", "w") as f:
                json.dump(best_config, f, indent=4)

    print("\n=== Filtering Complete ===")
    if best_config:
        print(f"Best configuration: Depth {best_config['depth']} with LER {best_config['total_ler']:.3e}")
        print(f"Saved securely in {DEFAULT_BEST_DIR}")
    else:
        print("No valid circuits were generated.")
    return out_dir, run_id, n_qubits, logical_qubits

##################
# Filtering Data #
##################

def agglomerate_results(json_dir: Path) -> Path | None:
    """
    Gathers all individual step results into one summary file.
    """
    data_points = []
    
    for file_path in json_dir.glob("step_*_results.json"):
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                if 'total_ler' in data:
                    data['step_name'] = file_path.name.replace("_results.json", "")
                    data_points.append(data)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {file_path.name}")

    if not data_points:
        print("No valid result files found to agglomerate.")
        return None

    # Sort Descending by LER
    data_points.sort(key=lambda x: x['total_ler'], reverse=True)

    # Save to a single master file in the run directory (one level up from json_dir)
    summary_file = json_dir.parent / "agglomerated_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(data_points, f, indent=4)
        
    print(f"\nSaved agglomerated results to {summary_file}")
    return summary_file