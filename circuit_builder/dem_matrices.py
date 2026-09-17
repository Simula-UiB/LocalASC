from apply_noise import add_circuit_level_noise
import stim
import numpy as np
from scipy.sparse import csc_matrix
import os
import argparse


def dem_matrices_saver(circuit_filepath: str) -> tuple[csc_matrix, csc_matrix]:
    circuit = stim.Circuit.from_file(circuit_filepath)

    # Assuming add_circuit_level_noise is defined elsewhere in your code
    noisy_circuit, _ = add_circuit_level_noise(circuit)

    dem = noisy_circuit.detector_error_model()

    # Flatten the DEM to expand any repeat blocks from the circuit
    flattened_dem = dem.flattened()

    # Get the dimensions of matrices
    num_detectors = dem.num_detectors
    num_observables = dem.num_observables

    # Fault mechanisms
    num_errors = sum(1 for inst in flattened_dem if inst.type == 'error')

    # Lists to store the (row, column) coordinates for our sparse matrices
    H_rows, H_cols = [], []
    L_rows, L_cols = [], []

    error_idx = 0
    for inst in flattened_dem:
        if inst.type == 'error':
            for target in inst.targets_copy(): 
                if target.is_relative_detector_id():
                    H_rows.append(target.val)
                    H_cols.append(error_idx)
                elif target.is_logical_observable_id():
                    L_rows.append(target.val)
                    L_cols.append(error_idx)
            
            # Move to the next column for the next independent error mechanism
            error_idx += 1

    # Build sparse matrices
    H_dem = csc_matrix((np.ones(len(H_rows), dtype=np.uint8), (H_rows, H_cols)), 
                       shape=(num_detectors, num_errors))
    
    L = csc_matrix((np.ones(len(L_rows), dtype=np.uint8), (L_rows, L_cols)), 
                   shape=(num_observables, num_errors))

    return H_dem, L

def save_to_alist(sparse_matrix, filepath: str):
    """
    Saves a scipy.sparse matrix to the standard AList format.
    AList uses 1-based indexing and pads rows/cols with 0s up to the max weight.
    """
    # We need both CSR and CSC
    mat_csr = sparse_matrix.tocsr()
    mat_csc = sparse_matrix.tocsc()
    
    num_rows, num_cols = sparse_matrix.shape
    
    # Calculate weights 
    row_weights = np.diff(mat_csr.indptr)
    col_weights = np.diff(mat_csc.indptr)
    
    max_row_weight = row_weights.max() if num_rows > 0 else 0
    max_col_weight = col_weights.max() if num_cols > 0 else 0
    
    with open(filepath, 'w') as f:
        # 1. Dimensions: cols, rows
        f.write(f"{num_cols} {num_rows}\n")
        
        # 2. Max weights: max_col_weight, max_row_weight
        f.write(f"{max_col_weight} {max_row_weight}\n")
        
        # 3. Column weights list
        f.write(" ".join(map(str, col_weights)) + "\n")
        
        # 4. Row weights list
        f.write(" ".join(map(str, row_weights)) + "\n")
        
        # 5. Non-zero row indices for each column (1-based, padded with 0)
        for i in range(num_cols):
            start, end = mat_csc.indptr[i], mat_csc.indptr[i+1]
            indices = mat_csc.indices[start:end] + 1  # +1 for 1-based indexing
            padded = list(indices) + [0] * (max_col_weight - len(indices))
            f.write(" ".join(map(str, padded)) + "\n")
            
        # 6. Non-zero column indices for each row (1-based, padded with 0)
        for i in range(num_rows):
            start, end = mat_csr.indptr[i], mat_csr.indptr[i+1]
            indices = mat_csr.indices[start:end] + 1  # +1 for 1-based indexing
            padded = list(indices) + [0] * (max_row_weight - len(indices))
            f.write(" ".join(map(str, padded)) + "\n")


def save_to_txt(sparse_matrix, filepath: str):
    """
    Saves a scipy.sparse matrix to a dense standard text file.
    WARNING: For very large DEMs, converting to a dense array may cause memory issues.
    """
    dense_matrix = sparse_matrix.toarray().astype(np.uint8)
    # %d formats it as integers (0 or 1) rather than floats
    np.savetxt(filepath, dense_matrix, fmt='%d', delimiter=' ')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract H and L matrices from a Stim circuit DEM.")
    parser.add_argument("filepath", type=str, help="Path to the .stim circuit file.")
    args = parser.parse_args()

    # Get the base filename without extension
    base_name = os.path.splitext(os.path.basename(args.filepath))[0]
    
    # 1. Get the absolute path of the directory containing the circuit file
    circuit_dir = os.path.dirname(os.path.abspath(args.filepath))
    
    # 2. Go one level up to find the project root directory
    root_dir = os.path.dirname(circuit_dir)
    
    # 
    out_dir = os.path.join(root_dir, "results")
    os.makedirs(out_dir, exist_ok=True)

    txt_dir = os.path.join(out_dir, "txt")
    os.makedirs(txt_dir, exist_ok=True)

    alist_dir = os.path.join(out_dir, "alist")
    os.makedirs(alist_dir, exist_ok=True)

    print(f"Processing {args.filepath}...")
    H_dem, L = dem_matrices_saver(args.filepath)

    print(f"Extracted H_dem: {H_dem.shape[0]} detectors x {H_dem.shape[1]} errors")
    print(f"Extracted L: {L.shape[0]} observables x {L.shape[1]} errors")

    # Define output paths strictly inside the root/results folder
    h_alist_path = os.path.join(alist_dir, f"{base_name}_H_dem.alist")
    l_alist_path = os.path.join(alist_dir, f"{base_name}_L.alist")
    h_txt_path = os.path.join(txt_dir, f"{base_name}_H_dem.txt")
    l_txt_path = os.path.join(txt_dir, f"{base_name}_L.txt")

    print(f"Saving files to {out_dir}/ ...")
    save_to_alist(H_dem, h_alist_path)
    save_to_alist(L, l_alist_path)
    save_to_txt(H_dem, h_txt_path)
    save_to_txt(L, l_txt_path)

    print("Done! Files saved successfully.")