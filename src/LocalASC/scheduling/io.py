import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_SCHEDULE_DIR = PROJECT_ROOT / "results" / "schedules"

################
# CSV exporter #
################

def export_schedule_csv(filename: str | Path, lift_validate_output: tuple, output_dir: str | Path = "results/schedules" ) -> None:
    """
    Saves the lifted full schedule (X and Z parity checks) to a CSV format.
    """
    # Outputs
    TX, TZ, conf, bad, pairs, lift_dt, ne = lift_validate_output
    
    # Print validation stats to the console
    print(f"lift: {ne} edges, {conf} distinctness conflicts, {bad}/{pairs} parity violations "
          f"(both must be 0), {lift_dt * 1000:.3f} ms")
          
    # Destination path
    safe_filename = Path(filename).name
    filepath = Path(output_dir) / safe_filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Write the CSV
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("type,check,data_qubit,tick\n")
        for r in sorted(TX):
            for q in sorted(TX[r]): 
                f.write(f"X,{r},{q},{TX[r][q]}\n")
        for r in sorted(TZ):
            for q in sorted(TZ[r]): 
                f.write(f"Z,{r},{q},{TZ[r][q]}\n")
                
    print(f"wrote {filepath}")
