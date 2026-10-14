import argparse, sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.LocalASC.simulation.circuit_filtering import run_filtering_pipeline, agglomerate_results
from src.LocalASC.visualization.processing_and_plotting import plot_filtering_results

#To modify decoder parameters, edit the call to evaluate_circuit_fast in circuit_filtering

def main():
    ap = argparse.ArgumentParser(description="Run the circuit filtering pipeline to find the best schedule.")
    
    # Parity Check Matrices for CSS codes
    ap.add_argument("hx_path", type=str, help="Path to the Hx parity check matrix file (.txt)")
    ap.add_argument("hz_path", type=str, help="Path to the Hz parity check matrix file (.txt)")
    
    # Scheduling & Structural Parameters
    ap.add_argument("--rounds", type=int, default=3, help="Number of syndrome extraction rounds (default: 3)")
    ap.add_argument("-D", "--target_depth", type=int, default=None, help="Target depth for the schedule (default: Tanner Graph max_weight)")
    ap.add_argument("-s", "--steps", dest="filtering_steps", type=int, default=20, help="Number of schedules to search (default: 20)")
    ap.add_argument("-t", "--max_time_hrs", type=float, default=2.0, help="Maximum pipeline runtime in hours (default: 2.0)")
    
    # Evaluation Parameters
    ap.add_argument("-p", "--eval_p", type=float, default=0.001, help="Physical error rate for filtering evaluation (default: 0.001)")
    ap.add_argument("--shots", type=int, default=10000, help="Max shots per evaluation (default: 10000)")
    ap.add_argument("--max_errors", type=int, default=50, help="Target errors per evaluation (default: 50)")

    #Visualization of filtering
    ap.add_argument("--plot", action="store_true", help="Plot the agglomerated results after filtering completes")
    ap.add_argument("--save_plot", action="store_true", help="Save the plot as a JPG to results/plots/filtering/")
    
    args = ap.parse_args()

    # Load the matrices
    print(f"Loading matrices from {args.hx_path} and {args.hz_path}...")
    hx = np.loadtxt(args.hx_path, dtype=int)
    hz = np.loadtxt(args.hz_path, dtype=int)

    if args.target_depth is None:
        # Compute the max weight of the Tanner graph
        hx_max_row_weight = np.max(np.sum(hx, axis=1))
        hx_max_col_weight = np.max(np.sum(hx, axis=0))
        hx_max_weight = max(hx_max_col_weight, hx_max_row_weight)

        hz_max_row_weight = np.max(np.sum(hz, axis=1))
        hz_max_col_weight = np.max(np.sum(hz, axis=0))
        hz_max_weight =  max(hz_max_col_weight, hz_max_row_weight)

        max_tg_weight = max(hx_max_weight, hz_max_weight)
        args.target_depth = max_tg_weight
    
    # Execute the filtering pipeline
    run_dir, run_id, n, k = run_filtering_pipeline(
        Hx=hx, 
        Hz=hz, 
        rounds=args.rounds, 
        max_time_hrs=args.max_time_hrs, 
        eval_p=args.eval_p, 
        eval_shots=args.shots, 
        eval_errors=args.max_errors, 
        target_depth=args.target_depth, 
        number_of_searched_circs=args.filtering_steps
    )

    if run_dir:
        json_dir = run_dir / "schedule_json"
        summary_file = agglomerate_results(json_dir=json_dir)
        
        # Plot only and save
        if (args.plot or args.save_plot) and summary_file:
            save_path = None
            if args.save_plot:
                plot_dir = PROJECT_ROOT / "results" / "plots" / "filtering"
                
                plot_dir.mkdir(parents=True, exist_ok=True)
                
                save_path = plot_dir / f"filtering_run{run_id}_n{n}_k{k}.jpg"
            
            plot_filtering_results(summary_file=summary_file, save_path=save_path)

if __name__ == "__main__":
    main()