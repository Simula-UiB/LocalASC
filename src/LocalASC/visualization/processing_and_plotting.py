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
    parse the JSON lines output, aggregate by p and compute LER.
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

def filter_min_errors(df, min_errors):
    """
    keep rows with at least `min_errors` observed logical failures, summed over
    the X- and Z-memory experiments
    """
    if min_errors is None:
        return df
    return df[(df['errors_x'] + df['errors_z']) >= min_errors]

def get_fit_params(df, k, d_circ):
    """
    obtains fit parameters as described in Section VI of the paper
    df is the data frame, k is the code dimension and d_circ is the circuit distance (can also be estimation)
    """
    # Filters out pL = 0 points for the fit
    df_valid = df[df['total_pL'] > 0].copy()
    if len(df_valid) < 3:
        return None

    log_p = np.log(df_valid['p'].values)
    pL = df_valid['total_pL'].values
    pL_err = df_valid['total_pL_err'].values
    
    log_pL_err = np.maximum(pL_err, 1e-15) / pL
    weights = 1.0 / log_pL_err
    
    poly_coeffs = np.polyfit(log_p, np.log(pL), 2, w=weights)
    poly = np.poly1d(poly_coeffs)
    
    p_ps = fsolve(lambda p: np.exp(poly(np.log(p))) - (1 - (1 - p)**k), 0.0055)[0]
    
    df_sub = df_valid[df_valid['p'] < p_ps]
    if len(df_sub) < 3:
        return None
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

def export_pgfplots_data(filepath, d_circ, name, out_dir, min_errors: int | None = 5,
                         x_range=(9e-5, 0.01), n_fit_points=200):
    """
    Write the data points and the extrapolation curve as pgfplots CSV files,
    using the same logic as plot_qec_results():
        <out_dir>/<name>_data.csv : p, total_pL, total_pL_err
        <out_dir>/<name>_fit.csv  : p, pL
    """
    df, k, d = load_and_process_json(filepath)
    df = filter_min_errors(df, min_errors)
    fit_params = get_fit_params(df, k, d_circ)
    if fit_params is None:
        raise ValueError(f"{name}: too few points with at least {min_errors} errors to fit.")
    p_ps, coeffs, exp = fit_params

    x_min, x_max = x_range
    df_data = df[(df['p'] >= x_min) & (df['p'] <= x_max) & (df['total_pL'] > 0)]
    data = pd.DataFrame({
        'p': df_data['p'],
        'total_pL': df_data['total_pL'],
        'total_pL_err': df_data['total_pL_err'],
    })

    sub_threshold_points = df[df['p'] < p_ps]['p']
    stop_line_p = sub_threshold_points.max() if not sub_threshold_points.empty else p_ps
    p_fit = np.logspace(np.log10(x_min), np.log10(stop_line_p), n_fit_points)
    fit = pd.DataFrame({'p': p_fit, 'pL': extrap(p_fit, coeffs, exp)})

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_dir / f"{name}_data.csv", index=False, float_format='%.6e')
    fit.to_csv(out_dir / f"{name}_fit.csv", index=False, float_format='%.6e')
    print(f"Wrote {out_dir / f'{name}_data.csv'} and {out_dir / f'{name}_fit.csv'}")
    return data, fit

######################
# Plotting functions #
######################

def plot_qec_results(file_configs: list, save_path: str | Path | None = None,
                     min_errors: int | None = 5, title: str = 'Memory Experiment Results'):
    """
    Plot the results from run_sinter_experiment()
    """
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(10, 8))

    x_min, x_max = 9e-5, 0.01
    p_fine = np.logspace(np.log10(x_min), np.log10(x_max), 1000)

    for filepath, color, marker, lbl, d_circ in file_configs:
        df, k, d = load_and_process_json(filepath)
        if df.empty:
            continue

        # low-statistics cut for the fit and for the plotted points
        df = filter_min_errors(df=df , min_errors=min_errors)
        if df.empty:
            print(f"{lbl}: no points with at least {min_errors} errors, nothing to plot.")
            continue

        df_plot = df[(df['p'] >= x_min) & (df['p'] <= x_max)].copy()

        # Plot the scattered data points with error bars
        plt.errorbar(df_plot['p'], df_plot['total_pL'], yerr=df_plot['total_pL_err'], 
                     fmt=marker, color=color, capsize=3, alpha=0.8, label=lbl, markersize=6)
        
        fit_params = get_fit_params(df, k, d_circ)
        if fit_params is None:
            print(f"{lbl}: too few points with at least {min_errors} errors, skipping the fit.")
            continue
        p_ps, coeffs, exp = fit_params

        # Plot Fitted Sub-threshold Curve
        sub_threshold_points = df[df['p'] < p_ps]['p']
        stop_line_p = sub_threshold_points.max() if not sub_threshold_points.empty else p_ps
            
        p_fine_sub = p_fine[p_fine <= stop_line_p]
        plt.plot(p_fine_sub, extrap(p_fine_sub, coeffs, exp), '-.', color=color, linewidth=2)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Physical Error Rate ($p$)', fontsize=14)
    plt.ylabel('Logical Error Rate ($p_L$)', fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()

    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, format='jpg', dpi=300, bbox_inches='tight')
        print(f"Saved plot image to {save_path}")

    plt.show()

def plot_filtering_results(summary_file: Path, save_path: Path | None = None) -> None:
    """
    Reads the agglomerated summary file and plots the LER trend.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed. Skipping plot.")
        return

    # Load the sorted data
    with open(summary_file, 'r') as f:
        data_points = json.load(f)

    if not data_points:
        return

    names = [pt['step_name'] for pt in data_points]
    lers = np.array([pt['total_ler'] for pt in data_points])
    
    ler_errs = np.array([pt.get('total_ler_err', 0.0) for pt in data_points])
    x_positions = np.array(range(len(names))) + 1

    # Asymmetric bars: keep the lower end strictly positive for the log axis
    yerr = np.vstack([np.minimum(ler_errs, lers * (1 - 1e-3)), ler_errs])

    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(12, 6))

    plt.bar(x_positions, lers, color='#4C72B0', alpha=0.6, edgecolor='black', label='LER (Bar)')
    plt.errorbar(x_positions, lers, yerr=yerr, fmt='none', ecolor='black',
                 elinewidth=1.2, capsize=4, label='LER $\\pm 1\\sigma$')
    plt.plot(x_positions, lers, color='#C44E52', marker='o', linewidth=2, markersize=6, label='LER Trend')

    if not np.any(ler_errs > 0):
        print("No 'total_ler_err' found in summary; plotting without error bars.")

    plt.xticks(x_positions, rotation=45, ha="right")
    plt.yscale('log')
    plt.ylabel('Logical Error Rate ($p_L$) [Log Scale]', fontsize=14)
    plt.xlabel('Candidate Circuit', fontsize=14)
    plt.title('Circuit Filtering Results (Ranked by LER)', fontsize=16, pad=15)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()
    
    # Save the plot if path is provided
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, format='jpg', dpi=300, bbox_inches='tight')
        print(f"Saved plot image to {save_path}")

    plt.show()
