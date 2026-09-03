import os
import sys
import multiprocessing

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import json
from ldpc.bposd_decoder import BpOsdDecoder
from scipy.sparse import csc_matrix
import time
from circuit_builder.local_ASC_scheduler import add_circuit_level_noise
from ldpc.sinter_decoders.sinter_bposd_decoder import SinterBpOsdDecoder
import sinter
import argparse
import stim
import matplotlib.pyplot as plt




def dem_to_matrices(dem):
    cols = {}
    for inst in dem.flattened():
        if inst.type != 'error': continue
        p = inst.args_copy()[0]
        dets = set(); obs = set()
        for t in inst.targets_copy():
            if t.is_relative_detector_id(): dets.add(t.val)
            elif t.is_logical_observable_id(): obs.add(t.val)
        key = (frozenset(dets), frozenset(obs))
        if key in cols: cols[key] = cols[key]*(1-p) + p*(1-cols[key])
        else: cols[key] = p
    nd, no = dem.num_detectors, dem.num_observables
    keys = list(cols)
    ri, ci, oi, oc, pr = [], [], [], [], []
    for j, (ds, os_) in enumerate(keys):
        for dd in ds: ri.append(dd); ci.append(j)
        for oo in os_: oi.append(oo); oc.append(j)
        pr.append(cols[(ds, os_)])
    H = csc_matrix((np.ones(len(ri), np.uint8), (ri, ci)), shape=(nd, len(keys)))
    O = csc_matrix((np.ones(len(oi), np.uint8), (oi, oc)), shape=(no, len(keys)))
    return H, O, np.array(pr)

def run_bposd_decoder(circ,shots,seed=42):
    dem = circ.detector_error_model(decompose_errors=False)
    H, O, pr = dem_to_matrices(dem)
    dec = BpOsdDecoder(H, error_channel=list(pr), max_iter=20,
                        bp_method='minimum_sum', ms_scaling_factor=0.625,
                        osd_method='OSD_0', osd_order=0)
    sampler = circ.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots, separate_observables=True)
    Od = O.toarray().astype(np.uint8)
    nerr = 0; t0 = time.time()
    for s in range(shots):
            e = dec.decode(dets[s].astype(np.uint8))
            pred = (Od @ e) % 2
            if np.any(pred != obs[s]): nerr += 1
    dt = time.time()-t0
    return nerr, shots, dt, H.shape

def sinter_decoding_experiment(circ, phys_err_probs, rounds, shots, code_distance, output_filename, basis='Z', name='Code', max_errors=10000, max_iter=100, seed=42):
    sinter_tasks = []
    
    for p in phys_err_probs:

        noisy_circ = add_circuit_level_noise(circ, p1q=p, p2q=p, p_meas=p)
        
        dem = noisy_circ.detector_error_model(decompose_errors=False)
        shp = (dem.num_detectors, dem.num_observables)
        
        task = sinter.Task(
                circuit=noisy_circ,
                json_metadata={
                   'p': p,                  
                   'd': code_distance,                
                   'rounds': rounds,
                   'shp': shp,
                   'shots': shots
                }
            )

        sinter_tasks.append(task)
    
    stats = sinter.collect(
        num_workers=multiprocessing.cpu_count(),               
        tasks=sinter_tasks,       
        decoders=["bposd"],          
        max_errors=max_errors, 
        max_shots=shots,           
        custom_decoders={
            "bposd": SinterBpOsdDecoder(
                max_iter=max_iter, 
                bp_method="ms", 
                osd_method="osd0"
            )
        }
    )

    for stat in stats:
        p = stat.json_metadata['p']
        shp = stat.json_metadata['shp']
        nerr = stat.errors
        ns = stat.shots
        
        dt = stat.seconds 
        
        ler = (nerr / ns) if ns > 0 else 0.0
        
        rec = dict(name=name, p=p, shots=ns, errors=nerr, seed=seed, basis=basis, decode_s=round(dt, 1))
        
        # Write to the dynamically generated filename
        with open(output_filename, 'a') as f: 
            f.write(json.dumps(rec) + '\n')
            
        print(f"{name} mem-{basis} p={p}: {nerr}/{ns} logical errors (LER={ler:.4g}, decode {dt:.0f}s, DEM {shp})")

def plot_multiple_jsons(filenames: list, output_image: str = "comparison_plot.png"):
    plt.figure(figsize=(9, 6))
    
    # Define distinct styles for your expected circuit types
    style_map = {
        "ATB": {"color": "blue", "marker": "o"},
        "localASC": {"color": "green", "marker": "s"},
        "fullASC": {"color": "red", "marker": "^"}
    }

    for filename in filenames:
        if not os.path.exists(filename):
            print(f"Warning: File {filename} not found. Skipping.")
            continue
            
        plot_data = {'p': [], 'ler': []}
        label_name = "Unknown"
        
        with open(filename, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                    
                record = json.loads(line)
                p = record['p']
                errors = record['errors']
                shots = record['shots']
                label_name = record.get('name', 'Code') # ATB, localASC, or fullASC
                
                ler = errors / shots if shots > 0 else 0
                if ler > 0: # Filter out 0 errors for log scale
                    plot_data['p'].append(p)
                    plot_data['ler'].append(ler)
        
        # Sort points by physical error rate
        sorted_points = sorted(zip(plot_data['p'], plot_data['ler']))
        if not sorted_points:
            continue
            
        x_p = [pt[0] for pt in sorted_points]
        y_ler = [pt[1] for pt in sorted_points]
        
        # Apply specific styling if it matches your expected names
        style = style_map.get(label_name, {"color": "black", "marker": "x"})
        
        # Add 'd' (distance) or other params to the label if you extract it from filename
        # Assuming filename convention: name_n{n}_k{k}_d{d}_nshots{shots}.json
        parts = os.path.basename(filename).split('_')
        d_val = next((part for part in parts if part.startswith('d')), "d?")
        display_label = f"{label_name} ({d_val})"
        
        plt.plot(x_p, y_ler, label=display_label, color=style["color"], marker=style["marker"], linestyle='-')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Physical Error Rate (p)', fontsize=12)
    plt.ylabel('Logical Error Rate (LER)', fontsize=12)
    plt.title('Code Comparison: ATB vs localASC vs fullASC', fontsize=14)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_image, dpi=300)
    print(f"Plot saved successfully as: {output_image}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    phys_err_probs = np.logspace(-5, -2, 20)

    # Required parameters
    parser.add_argument("-n", type=int, required=True, help="Number of physical qubits (n)")
    parser.add_argument("-k", type=int, required=True, help="Number of logical qubits (k)")
    parser.add_argument("-d", type=int, required=True, help="Code distance (d)")
    parser.add_argument("-name", type=str, required=True, help="Select a type of circuit from ATB, localASC and fullASC")

    # Optional parameters
    parser.add_argument("-shots", type=int, required=False, default=1000, help="Number of shots")
    parser.add_argument("-max_errors", type=int, required=False, default=1000, help="Maximum number of errors")
    parser.add_argument("-max_iter", type=int, required=False, default=100, help="Maximum number of iterations")
    parser.add_argument("-seed", type=int, required=False, default=42, help="Random seed")
    parser.add_argument("-basis", type=str, default='Z', help="Basis for memory experiment")

    parser.add_argument("--debug", action="store_true", help="Debug mode")

    parser.add_argument("--plot_only", action="store_true", help="Skip the simulation and only plot the existing JSON file")

    args = parser.parse_args()
    x = np.random.randint(0,1000)
    # 
    custom_filename = f"{args.name}_n{args.n}_k{args.k}_d{args.d}_nshots{args.shots}.json"

    if args.plot_only:
        custom_filenames = [f'{name}_n{args.n}_k{args.k}_d{args.d}_nshots{args.shots}.json' for name in ['ATB_official', 'localASC', 'fullASC']]
        print(f"Plotting only mode activated. Looking for files")
        print(custom_filenames)
        # Make sure the plot_multiple_jsons function from the previous step is defined above
        plot_multiple_jsons(custom_filenames)
        
    else:
        # Circuit Directions
        if args.n == 48:
            folder_name = "Codes/48_4_Coset2BGA/circuits/"
            if args.name == 'ATB_official':
                depth = 7
            else:
                depth = 6

            file_direction = folder_name + f'48_4_Coset2BGA_circuit_{args.name}_depth{depth}_mem{args.basis}.stim'
        else:
            folder_name = f"Codes/BB_codes/BB_{args.n}_{args.k}_{args.d}/circuits/"

            if args.name == 'ATB':
                file_direction = folder_name + f'BB_{args.n}_{args.k}_{args.d}_ATB_official_depth7_mem{args.basis}.stim'
            elif args.name == 'localASC':
                file_direction = folder_name + f'BB_{args.n}_{args.k}_{args.d}_localASC_depth7_mem{args.basis}.stim'
            elif args.name == 'fullASC':
                file_direction = folder_name + f'BB_{args.n}_{args.k}_{args.d}_fullASC_depth7_mem{args.basis}.stim'

        circ = stim.Circuit.from_file(file_direction)
        if args.debug:
            print(f"Number of observables: {circ.num_observables}")

            circ = add_circuit_level_noise(circ, 0.00001, 0.00001, 0.00001)

            dem = circ.detector_error_model(decompose_errors=False)
            H, O, pr = dem_to_matrices(dem)
            print(H.shape)
            print(O.shape)
            print(f"The matrices are orthogonal {O @ H.T}")


        else:
            rounds = args.d # Assuming rounds equals distance, adjust if necessary

            print(f"Starting decoding experiment with {args.name} circuit, {args.shots} shots per error level")
            sinter_decoding_experiment(
                circ=circ,
                phys_err_probs=phys_err_probs,
                rounds=rounds,
                shots=args.shots,
                code_distance=args.d,
                output_filename=custom_filename,
                name=args.name,
                max_errors=args.max_errors,
                max_iter=args.max_iter,
                seed=args.seed
            )

     