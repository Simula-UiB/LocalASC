import argparse
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.LocalASC.scheduling.profile_scheduler import profile_schedule_generation

# Standard codes for the table reproduction
CODES = {
    "LLSC25_12_4_d2": None, "LLSC25_24_4_d4": None, "LLSC25_28_6_d4": None,
    "PK25_36_4_d4": None, "Wang-etal26_36_4_d6": None,
    "LLSC25_48_4_d8": None, "ATB_48_4_d8": None,
    "LP23_84_14_d10": None, "LP23_96_12_d10": None, "WM24_108_12_d6": None,
    "LP23_112_12_d12": None, "LP23_120_8_d8": None,
    "45_7_35_d4": ("Hx_45_7_35_d4_completed.txt", "Hz_45_7_35_d4_completed.txt"),
    "WLLKG26_96_30_d4": None, "WLLKG26_128_40_d4": None, "WLLKG26_150_62_d4": None,
    "180_26_d6": ("GenTannerSch_180_26_d6_2173_X_completed.txt", "GenTannerSch_180_26_d6_2173_Z_completed.txt"),
    "Tan2TNCb_325_14_d9_5": ("Tan2TNCb_325_14_d9_5_Hx_completed.txt", "Tan2TNCb_325_14_d9_5_Hz_completed.txt"),
    "Tan2TNCb_666_4_65": None, "Tan2TNCb_666_8_68": None,
}

POLICY_MAP = {"PK25_36_4_d4": "odd", "Wang-etal26_36_4_d6": "odd", "180_26_d6": "full"}

def main():
    parser = argparse.ArgumentParser(description="Reproduce runtime results for Table I.")
    
    # Core settings
    parser.add_argument("--policy", type=str, default="hybrid", choices=["hybrid", "ladder", "maxodd", "odd", "full"], help="Subgroup search policy")
    parser.add_argument("--codes_dir", type=str, default=str(PROJECT_ROOT / "codes"), help="Directory containing code matrices")
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "results" / "timing_tables" /"localasc_table_timing.jsonl"), help="Output JSONL file path")
    parser.add_argument("--runs", type=int, default=10, help="Number of profiling runs per code")
    
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
    
    for code in target_codes:
        code_folder = codes_dir / code
        fx, fz = CODES[code] or (f"{code}_Hx.txt", f"{code}_Hz.txt")
        hx_path = code_folder / fx
        hz_path = code_folder / fz
        
        if not hx_path.exists() or not hz_path.exists():
            print(f"[Skip] Matrix files for {code} not found in {code_folder}")
            continue
            
        Hx = np.loadtxt(hx_path, dtype=int)
        Hz = np.loadtxt(hz_path, dtype=int)
        
        current_policy = args.policy if args.policy in ("ladder", "hybrid") else POLICY_MAP.get(code, "maxodd")
        files_metadata = [fx, fz]
        
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

if __name__ == "__main__":
    main()