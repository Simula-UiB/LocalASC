import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from pathlib import Path

##############################
# Experiment data processing #
##############################

def load_and_process_json(filepath):
    """
    Parses the JSON lines output, aggregates by p and computes LER.
    """
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            
    df = pd.DataFrame(rows)
    if df.empty:
        return df, None, None
    
    if 'p' in df.columns:
        df['p'] = df['p'].round(10)
        
    df = df.groupby('p', as_index=False).agg({
        'shots_x': 'sum',
        'errors_x': 'sum',
        'shots_z': 'sum',
        'errors_z': 'sum',
        'd': 'first',
        'k': 'first'
    })

    # Standardize column names for the fitting function
    d = df['d'].iloc[0]
    k = df['k'].iloc[0]

    # Calculate raw block error rates
    raw_x = df['errors_x'] / df['shots_x']
    raw_z = df['errors_z'] / df['shots_z']
    
    # Safe variables to prevent division by zero or negative fractional roots
    safe_raw_x = np.minimum(raw_x, 1.0 - 1e-10)
    safe_raw_z = np.minimum(raw_z, 1.0 - 1e-10)
    
    # Per-round logical error rates
    df['pL_x'] = 1 - (1 - safe_raw_x)**(1/d)
    df['pL_z'] = 1 - (1 - safe_raw_z)**(1/d)
    
    # Base standard errors for the raw block rates (binomial distribution)
    safe_errors_x = np.maximum(df['errors_x'], 1)
    safe_p_fail_x = safe_errors_x / df['shots_x']
    err_x = np.sqrt(safe_p_fail_x * (1 - safe_p_fail_x) / df['shots_x'])

    safe_errors_z = np.maximum(df['errors_z'], 1)
    safe_p_fail_z = safe_errors_z / df['shots_z']
    err_z = np.sqrt(safe_p_fail_z * (1 - safe_p_fail_z) / df['shots_z'])
    
    # Propagate error
    df['pL_err_x'] = (1/d) * (1 - safe_raw_x)**(1/d - 1) * err_x
    df['pL_err_z'] = (1/d) * (1 - safe_raw_z)**(1/d - 1) * err_z

    # The total aggregated LER
    df['total_pL'] = 1 - (1 - df['pL_x']) * (1 - df['pL_z'])
    
    # Propagate error to the total aggregated LER
    df['total_pL_err'] = np.sqrt(
        ((1 - df['pL_z']) * df['pL_err_x'])**2 + 
        ((1 - df['pL_x']) * df['pL_err_z'])**2
    )

    return df.sort_values('p'), k, d

def get_fit_params(df, k, d_circ):
    # Filters out pL = 0 points for the fit
    df_valid = df[df['total_pL'] > 0].copy()
    
    log_p = np.log(df_valid['p'].values)
    pL = df_valid['total_pL'].values
    pL_err = df_valid['total_pL_err'].values
    
    log_pL_err = np.maximum(pL_err, 1e-15) / pL
    weights = 1.0 / log_pL_err
    
    poly_coeffs = np.polyfit(log_p, np.log(pL), 2, w=weights)
    poly = np.poly1d(poly_coeffs)
    
    p_ps = fsolve(lambda p: np.exp(poly(np.log(p))) - (1 - (1 - p)**k), 0.0055)[0]
    
    df_sub = df_valid[df_valid['p'] < p_ps]
    p_s = df_sub['p'].values
    pL_s = df_sub['total_pL'].values
    pL_err_s = df_sub['total_pL_err'].values
    
    exponent = d_circ / 2.0
    y = np.log(pL_s) - exponent * np.log(p_s)
    
    y_err = np.maximum(pL_err_s, 1e-15) / pL_s
    weights_sub = 1.0 / y_err
    
    c2, c1, c0 = np.polyfit(p_s, y, 2, w=weights_sub)
    print(f"Computed pseudo-threshold: {p_ps}")
    return p_ps, (c0, c1, c2), exponent

def extrap(p, c, e): 
    return (p**e) * np.exp(c[0] + c[1]*p + c[2]*p**2)

######################
# Plotting functions #
######################

def plot_qec_results(file_configs):
    """
    Plot the results from run_sinter_experiment()
    """
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(10, 8)) 

    x_min, x_max = 9e-5, 0.01
    p_fine = np.logspace(np.log10(x_min), np.log10(x_max), 1000)

    max_rel_err = 0.5

    for filepath, color, marker, lbl, d_circ in file_configs:
        df, k, d = load_and_process_json(filepath)
        if df.empty:
            continue
            
        p_ps, coeffs, exp = get_fit_params(df, k, d_circ)
        
        # Error Filtering (Identical to export)
        df_plot = df[(df['p'] >= x_min) & (df['p'] <= x_max)].copy()
        
        safe_pL = np.maximum(df_plot['total_pL'], 1e-15)
        relative_error = df_plot['total_pL_err'] / safe_pL
        df_plot = df_plot[relative_error <= max_rel_err]
        
        # Plot the scattered data points with error bars
        plt.errorbar(df_plot['p'], df_plot['total_pL'], yerr=df_plot['total_pL_err'], 
                     fmt=marker, color=color, capsize=3, alpha=0.8, label=lbl, markersize=6)
        
        # Plot Fitted Sub-threshold Curve (Identical logic)
        sub_threshold_points = df[df['p'] < p_ps]['p']
        stop_line_p = sub_threshold_points.max() if not sub_threshold_points.empty else p_ps
            
        p_fine_sub = p_fine[p_fine <= stop_line_p]
        plt.plot(p_fine_sub, extrap(p_fine_sub, coeffs, exp), '-.', color=color, linewidth=2)

        
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(x_min, x_max)
    plt.ylim(1e-13, 1e-1)

    plt.title('Circuit Level Performance', fontsize=18, pad=15)
    plt.xlabel('Physical Error Rate ($p$)', fontsize=18)
    plt.ylabel('Logical Error Rate per Syndrome Cycle ($p_L$)', fontsize=18)

    plt.legend(loc='lower right', fontsize=16, framealpha=0.9, edgecolor='gray')
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout() 

    plt.show()

def plot_filtering_results(folder_path: str | Path):
    """
    Reads all *_results.json files in a directory, sorts them by LER descending, 
    and plots them as a combined bar and line chart.
    """

    folder = Path(folder_path)
    data_points = []

    # Extract Data
    for file_path in folder.glob("*_results.json"):
        # Extract the prefix
        step_name = file_path.name.replace("_results.json", "")
        
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                if 'total_ler' in data:
                    data_points.append({
                        'name': step_name,
                        'ler': data['total_ler']
                    })
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {file_path.name}")

    if not data_points:
        print(f"No valid result files found in {folder_path}")
        return

    # Sort Descending by LER
    data_points.sort(key=lambda x: x['ler'], reverse=True)

    names = [pt['name'] for pt in data_points]
    lers = [pt['ler'] for pt in data_points]
    x_positions = range(len(names))

    # Plotting
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(12, 6))
    
    # Overlay the bar chart and the connecting line
    plt.bar(x_positions, lers, color='#4C72B0', alpha=0.6, edgecolor='black', label='LER (Bar)')
    plt.plot(x_positions, lers, color='#C44E52', marker='o', linewidth=2, markersize=6, label='LER Trend')

    # Formatting
    plt.xticks(x_positions, names, rotation=45, ha="right")
    
    plt.yscale('log')
    plt.ylabel('Logical Error Rate ($p_L$) [Log Scale]', fontsize=14)
        
    plt.xlabel('Candidate Circuit', fontsize=14)
    plt.title('Circuit Filtering Results (Ranked by LER)', fontsize=16, pad=15)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()

    plt.show()