import argparse
import sys
from pathlib import Path
import numpy as np
import stim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.LocalASC.circuit_builder.circuit_builder import build_CSS_memory_circuit, rank2_rref
from src.LocalASC.simulation.sinter_experiment import run_sinter_experiment
from src.LocalASC.scheduling.localASC_scheduler import generate_schedule
from src.LocalASC.visualization.processing_and_plotting import plot_qec_results

def main():
    ap = argparse.ArgumentParser(description="Run a decoding experiment using bposd+cs (Or custom decoder) over selected or automatically generated circuit")
    
    # Parity Check Matrices for CSS codes
    ap.add_argument("hx_path", type=str, help="Path to the Hx parity check matrix file (.txt)")
    ap.add_argument("hz_path", type=str, help="Path to the Hz parity check matrix file (.txt)")
    ap.add_argument("rounds", type=int, help="Syndrome extraction rounds")
    
    # Optional parameters
    ap.add_argument("-d", type=int, default=None, help="Code distance (for metadata)")
    ap.add_argument("--name", type=str, default=None, help="Name of the code for metadata")
    
    # Call for custom circuits or schedule building
    ap.add_argument("-circuit_x", default=None, type=str, help="Path to memX circuit if specific circuit is desired")
    ap.add_argument("-circuit_z", default=None, type=str, help="Path to memZ circuit if specific circuit is desired")
    ap.add_argument("-schedule", default=None, type=str, help="Path to schedule CSV")
    ap.add_argument("-D", "--target_depth", type=int, default=None, help="Target depth to automatically generate a schedule")

    # Decoding parameters
    ap.add_argument("--shots", type=int, default=10000, help="Max shots per evaluation")
    ap.add_argument("--max_errors", type=int, default=100, help="Target errors per evaluation")
    ap.add_argument("--bpiter", type=int, default=10, help="Max BP iterations")
    ap.add_argument("-p", "--probs", nargs="+", type=float, default=[3e-3, 2e-3, 1e-3, 5e-4, 1e-4], help="List of physical error probabilities to evaluate")

    #Plot
    ap.add_argument("--plot", action="store_true", help="Plot the results after the experiment completes")
    ap.add_argument("--save_plot", action="store_true", help="Save the plot as a JPG to results/plots/sinter/")

    args = ap.parse_args()
    
    # Load matrices
    Hx = np.loadtxt(args.hx_path, dtype=int)
    Hz = np.loadtxt(args.hz_path, dtype=int)
    n = Hx.shape[1]
    kx, *_ = rank2_rref(Hx)
    kz, *_ = rank2_rref(Hz) 
    k = n-kx-kz

    code_name = args.name if args.name else f"Code_n{n}_k{k}"
    
    # Get circuits: load from file, build from provided schedule, or generate schedule from scratch
    if args.circuit_x and args.circuit_z:
        print("Loading circuits directly from provided files...")
        circ_x = stim.Circuit.from_file(args.circuit_x)
        circ_z = stim.Circuit.from_file(args.circuit_z)
        
    elif args.schedule:
        print(f"Building circuits dynamically from provided schedule: {args.schedule}...")
        circ_x = build_CSS_memory_circuit(Hx, Hz, args.schedule, "X", args.rounds)
        circ_z = build_CSS_memory_circuit(Hx, Hz, args.schedule, "Z", args.rounds)
        
    elif args.target_depth:
        print(f"Automatically generating schedule at depth {args.target_depth}...")
        
        # Define a temporary path for the generated schedule
        temp_schedule_path = PROJECT_ROOT / "results" / "schedules" / f"temp_n{n}_D{args.target_depth}.csv"
        temp_schedule_path.parent.mkdir(parents=True, exist_ok=True)
        
        success, status = generate_schedule(Hx, Hz, args.target_depth, str(temp_schedule_path), solver_tlim=120)
        
        if not success:
            print(f"Error: Schedule generation failed or timed out (Status: {status}). Try a different depth.")
            sys.exit(1)
            
        print(f"Schedule generated successfully. Building circuits...")
        circ_x = build_CSS_memory_circuit(Hx, Hz, str(temp_schedule_path), "X", args.rounds)
        circ_z = build_CSS_memory_circuit(Hx, Hz, str(temp_schedule_path), "Z", args.rounds)
        
    else:
        print("Error: You must provide either BOTH '-circuit_x' and '-circuit_z', OR a '-schedule', OR a '-D' target depth.")
        sys.exit(1)
        
    filename = f"n{n}_k{k}_shots{args.shots}_bpiter{args.bpiter}_rounds{args.rounds}.json"
    code_output_dir = PROJECT_ROOT / "results" / "sinter_data" / code_name
    filepath = code_output_dir / filename

    # Run the experiment
    run_sinter_experiment(
        base_circ_x=circ_x,
        base_circ_z=circ_z,
        physical_error_probs=args.probs,
        filename=filename,
        name=code_name,
        n=n,
        k=k,
        nrounds=args.rounds,
        num_shots=args.shots,
        max_errors=args.max_errors,
        max_bp_iter=args.bpiter,
        d=args.d,
        output_dir=code_output_dir
    )

    if args.plot or args.save_plot:
        # Format: (filepath, color, marker, label, distance)
        file_configs = [(filepath, '#4C72B0', 'o', code_name, args.d)]
        
        save_path = None
        if args.save_plot:
            save_path = PROJECT_ROOT / "results" / "plots" / "sinter" / code_name / f"n{n}_k{k}_shots{args.shots}_plot.jpg"
            
        print("Generating plot...")
        plot_qec_results(file_configs=file_configs, save_path=save_path)
    
if __name__ == "__main__":
    main()