import sys, argparse, os, json

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from codedistance import circuitDistance
import stim
from circuit_builder.local_ASC_scheduler import add_circuit_level_noise, add_phenomenological_noise



def compute_circuit_distance(circuit : stim.Circuit, exact_method : bool = False, noise_type : str = 'CLN'):
    """
    Given a noiseless circuit computes the circuit distance under the specified noise model (Circuit-Level or Phenomenological) using an exact or approximate method.
    Exact method is BZ algorithm and approximative is QdistRnDMW from webster's codedistance package.
    """
    #Placeholder for adding noise locations
    p = 0.00001

    if noise_type == 'CLN':
        n_circuit = add_circuit_level_noise(circuit,p,p,p)
    elif noise_type == 'PN':
        n_circuit = add_phenomenological_noise(circuit)

    
    #One can modify the parameters to fit the desired computation
    params = {
    'iterCount': 1000,     ## number of iterations - eg QDistRnd, decoderDist
    'maxTime': 3600*8,      ## max runtime for CLI methods (eg SAT, m4ri, )
    'nThreads': 8,          ## cater for multi-threaded methods
    'LOCheck': 0,            ## check logical operators
    'filterCircuit': False
    }

    #If using random algorithms
    seed = 42

    if exact_method:
        method = 'BZDistMW'
    else:
        method = 'QDistRndMW'

    code_dict = circuitDistance(n_circuit,method,params,seed)

    print(f"Circuit with numer of faults {code_dict['n']} has distance approximation {code_dict['d']}")

    return code_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute circuit distance for a specified QEC code.")
    
    # Required parameters
    parser.add_argument("-n", type=int, required=True, help="Number of physical qubits (n)")
    parser.add_argument("-k", type=int, required=True, help="Number of logical qubits (k)")
    parser.add_argument("-d", type=int, required=True, help="Code distance (d)")
    
    # Optional parameters
    default_codes_dir = os.path.join(project_root, "Codes")
    parser.add_argument("--dir", type=str, default=default_codes_dir, help="Directory containing the stim files")
    parser.add_argument("--prefix", type=str, default="code", help="Prefix for the file name (e.g., 'surface', 'color')")
    parser.add_argument("--noise", type=str, choices=['CLN', 'PN'], default='CLN', help="Noise model type")
    parser.add_argument("--exact", action="store_true", help="Use exact BZ algorithm instead of approximate")
    
    args = parser.parse_args()

    filename = f"{args.prefix}_n{args.n}_k{args.k}_d{args.d}.stim"
    filepath = os.path.join(args.dir, filename)

    if not os.path.exists(filepath):
        print(f"Error: Could not find circuit file at '{filepath}'")
        sys.exit(1)

    print(f"Loading circuit from: {filepath}")
    
    try:
        circuit = stim.Circuit.from_file(filepath)
    except Exception as e:
        print(f"Error loading Stim circuit: {e}")
        sys.exit(1)

    # Run the distance computation
    print(f"Computing distance under noise model ='{args.noise}' and exact_method= {args.exact}...")
    result = compute_circuit_distance(circuit, exact_method=args.exact, noise_type=args.noise)
    print("Done!")

    if args.exact:
        summary_msg = f"Found circuit distance = {result.get('d', 'N/A')}"
    else:
        summary_msg = f"Found circuit distance upper bound {result.get('d', 'N/A')}"

    print(summary_msg)
    
    base_filename = f"{args.prefix}_n{args.n}_k{args.k}_d{args.d}_{args.noise}"
    json_path = os.path.join(args.outdir, f"{base_filename}_results.json")
    txt_path = os.path.join(args.outdir, f"{base_filename}_summary.txt")
    
    with open(json_path, "w") as json_file:
        json.dump(result, json_file, indent=4)
        
    with open(txt_path, "w") as txt_file:
        txt_file.write(f"--- Circuit Distance Computation ---\n")
        txt_file.write(f"Circuit File: {filepath}\n")
        txt_file.write(f"Noise Model: {args.noise}\n")
        txt_file.write(f"Exact Method: {args.exact}\n")
        txt_file.write(f"------------------------------------\n")
        txt_file.write(f"{summary_msg}\n\n")
        txt_file.write(f"Full Result Dictionary:\n")
        txt_file.write(json.dumps(result, indent=4))
        
    print(f"\nResults successfully saved to:")
    print(f"  - Dictionary: {json_path}")
    print(f"  - Text Log:   {txt_path}")