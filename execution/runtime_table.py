import argparse
import sys, time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.LocalASC.scheduling.profile_scheduler import profile_schedule_generation

# Standard codes for the table reproduction
CODES = {
    "01_n12_k4_d2": ("01_n12_k4_d2_Hx.txt", "01_n12_k4_d2_Hz.txt"),             
    "02_n24_k4_d4": ("02_n24_k4_d4_Hx.txt", "02_n24_k4_d4_Hz.txt"),             
    "03_n28_k6_d4": ("03_n28_k6_d4_Hx.txt", "03_n28_k6_d4_Hz.txt"),             
    "04_n36_k4_d4": ("04_n36_k4_d4_Hx.txt", "04_n36_k4_d4_Hz.txt"),             
    "05_n36_k4_d6": ("05_n36_k4_d6_Hx.txt", "05_n36_k4_d6_Hz.txt"),             
    "06_n48_k4_d8": ("06_n48_k4_d8_Hx.txt", "06_n48_k4_d8_Hz.txt"),            
    "07_n48_k4_d8": ("07_n48_k4_d8_Hx.txt", "07_n48_k4_d8_Hz.txt"),             
    "08_n108_k12_d6": ("08_n108_k12_d6_Hx.txt", "08_n108_k12_d6_Hz.txt"),      
    "09_n84_k14_d10": ("09_n84_k14_d10_Hx.txt", "09_n84_k14_d10_Hz.txt"),      
    "10_n96_k12_d10": ("10_n96_k12_d10_Hx.txt", "10_n96_k12_d10_Hz.txt"),       
    "11_n112_k12_d12": ("11_n112_k12_d12_Hx.txt", "11_n112_k12_d12_Hz.txt"),    
    "12_n120_k8_d8": ("12_n120_k8_d8_Hx.txt", "12_n120_k8_d8_Hz.txt"),          
    "13_n45_k7_d4": ("13_n45_k7_d4_Hx.txt", "13_n45_k7_d4_Hz.txt"),             
    "14_n180_k26_d6": ("14_n180_k26_d6_Hx.txt", "14_n180_k26_d6_Hz.txt"),       
    "15_n325_k14_d9": ("15_n325_k14_d9_Hx.txt", "15_n325_k14_d9_Hz.txt"),       
    "16_n666_k4": ("16_n666_k4_Hx.txt", "16_n666_k4_Hz.txt"),                   
    "17_n666_k8": ("17_n666_k8_Hx.txt", "17_n666_k8_Hz.txt"),                   
    "18_n1225_k225_d6": ("18_n1225_k225_d6_Hx.txt", "18_n1225_k225_d6_Hz.txt"), 
}

POLICY_MAP = {"04_n36_k4_d4": "odd", "05_n36_k4_d6": "odd", "14_n180_k26_d6": "full"}

ALL_POLICIES = ["maxodd", "hybrid", "ladder", "full", "odd"]

def add_spaces_to_file(input_filepath, output_filepath):
    with open(input_filepath, 'r') as infile, open(output_filepath, 'w') as outfile:
        for line in infile:
            spaced_line = " ".join(line.strip()) + "\n"
            outfile.write(spaced_line)

def main():
    parser = argparse.ArgumentParser(description="Reproduce runtime results for Table I.")
    
    # Core settings
    parser.add_argument("--policy", type=str, default="hybrid", choices=["hybrid", "ladder", "maxodd", "odd", "full"], help="Subgroup search policy")
    parser.add_argument("--codes_dir", type=str, default=str(PROJECT_ROOT / "codes" / "Table1_PCMs"), help="Directory containing code matrices")
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "results" / "timing_tables" /"localasc_table_timing.jsonl"), help="Output JSONL file path")
    parser.add_argument("--runs", type=int, default=10, help="Number of profiling runs per code")
    parser.add_argument("--search_best", action="store_true", help="Cycle through all policies per code to find and assign the fastest one.")

    # Time limits
    parser.add_argument("--tlim", type=int, default=1800, help="Max time limit (s) for full searches (default: 1800)")
    parser.add_argument("--tlim_step", type=int, default=60, help="CP-SAT limit (s) per subgroup in ladder (default: 60)")
    parser.add_argument("--hybrid_cap", type=int, default=10, help="CP-SAT limit (s) on the first hybrid maxodd attempt (default: 10)")
    
    parser.add_argument("specific_codes", nargs="*", help="Specific code names to run (default: runs all in CODES)")
    
    args = parser.parse_args()
    
    codes_dir = Path(args.codes_dir)
    output_file = Path(args.output)
    
    target_codes = args.specific_codes if args.specific_codes else CODES.keys()
    
    print(f"=== Starting Runtime Table Generation ===")
    print(f"Codes Directory: {codes_dir}")
    print(f"Output File: {output_file}")

    best_policies_found = {}
    
    for code in target_codes:
        code_folder = codes_dir 
        fx, fz = CODES[code]
        hx_path = code_folder / fx
        hz_path = code_folder / fz
        
        if not hx_path.exists() or not hz_path.exists():
            print(f"[Skip] Matrix files for {code} not found in {code_folder}")
            continue
            
        Hx = np.loadtxt(hx_path, dtype=int)
        Hz = np.loadtxt(hz_path, dtype=int)
        print(f"Hx shape: {Hx.shape}, Hz shape: {Hz.shape}")
        files_metadata = [fx, fz]

        if args.search_best:
            print(f"\n--- Searching fastest policy for {code} ---")
            best_time = float('inf')
            best_policy = None

            for p in ALL_POLICIES:
                print(f"  -> Testing policy: {p}")
                start_t = time.time()
                
                try:
                    profile_schedule_generation(
                        code_name=code,
                        Hx=Hx,
                        Hz=Hz,
                        policy=p,
                        output_file=output_file,
                        runs=args.runs,
                        tlim=args.tlim,
                        tlim_step=args.tlim_step,
                        tlim_hybrid=args.hybrid_cap,
                        files_metadata=files_metadata
                    )
                    elapsed = time.time() - start_t
                    print(f"     [Success] Time taken: {elapsed:.2f}s")
                    
                    if elapsed < best_time:
                        best_time = elapsed
                        best_policy = p
                except Exception as e:
                    print(f"     [Failed] Policy '{p}' is incompatible or crashed: {e}")
            
            if best_policy:
                print(f"[*] Fastest policy for {code} is '{best_policy}' ({best_time:.2f}s)")
                best_policies_found[code] = best_policy
            else:
                print(f"[!] All policies failed for {code}.")

        else:
            current_policy = args.policy
            
            try:
                profile_schedule_generation(
                    code_name=code,
                    Hx=Hx,
                    Hz=Hz,
                    policy=current_policy,
                    output_file=output_file,
                    runs=args.runs,
                    tlim=args.tlim,
                    tlim_step=args.tlim_step,
                    tlim_hybrid=args.hybrid_cap,
                    files_metadata=files_metadata
                )
            except Exception as e:
                print(f"     [Failed] Policy '{current_policy}' failed for {code}: {e}")

    if args.search_best and best_policies_found:
        print("\n" + "="*50)
        print("SEARCH COMPLETE. NEW OPTIMAL POLICY MAP:")
        print("="*50)
        print("POLICY_MAP = {")
        for c, p in best_policies_found.items():
            print(f'    "{c}": "{p}",')
        print("}")

if __name__ == "__main__":
    main()