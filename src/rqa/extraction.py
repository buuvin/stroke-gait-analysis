import numpy as np
import time
import random
from pathlib import Path

from teaspoon.parameter_selection.MI_delay import MI_for_delay
from teaspoon.parameter_selection.FNN_n import FNN_n

from paths import SORTED_HEALTHY, SORTED_STROKE, EXTRACTED
from rqa.rqa_utils import find_opt_neighborhood

from config import RANDOM_SEED
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Fixed version of find_opt_params - uses corrected find_opt_neighborhood
def find_opt_params(data_file):
    """
    Find optimal RQA parameters for a single file.
    
    Args:
        data_file: path to the data file (Path object or string)
    
    Returns:
        tuple: (tau, n, neighborhood)
    """
    print(f"Processing: {data_file}")

    with open(data_file, 'r') as file:
        data_points = file.readlines()
    data_points = np.array(data_points, dtype=np.float64)

    # Simple z-score normalization to reduce file-by-file amplitude variation
    mean = np.mean(data_points)
    std = np.std(data_points)
    if std != 0:
        data_points = (data_points - mean) / std
    else:
        data_points = data_points * 0.0

    start = time.time()
    tau = MI_for_delay(data_points, method="kraskov 1", k=2)
    perFNN, n = FNN_n(data_points, tau, method='cao', maxDim=10)
    neighborhood = find_opt_neighborhood(data_points, n, tau)

    elapsed = time.time() - start
    print(f"  Done in {elapsed:.2f}s - τ={tau}, n={n}, neighborhood={neighborhood:.6f}")
    
    return (tau, n, neighborhood)

# Function to find and store optimal parameters for a specific group
def find_group_optimal_params(group_files, category, cop_type, eye_condition, affected_side="", axis=""):
    """
    Find optimal parameters for a group by averaging parameters from sampled files.
    
    Args:
        group_files: list of file paths for this group
        category: "healthy" or "stroke"
        cop_type: "COP1", "COP2", or "COP"
        eye_condition: "eyes_open" or "eyes_closed"
        affected_side: "" for healthy, "left_affected" or "right_affected" for stroke
        axis: "x" or "y"
    
    Returns:
        dict with averaged optimal parameters
    """
    # Sample up to 30 files (or use all if fewer)
    sample_size = min(30, len(group_files))
    sampled_files = random.sample(group_files, sample_size) if len(group_files) > sample_size else group_files
    
    print(f"\n{'='*60}")
    print(f"Processing group: {category}/{affected_side}/{cop_type}/{eye_condition}/{axis}")
    print(f"Total files: {len(group_files)}, Sampling: {len(sampled_files)}")
    print(f"{'='*60}")
    
    taus = []
    ns = []
    neighborhoods = []
    
    # Find optimal parameters for each sampled file
    for file_path in sampled_files:
        try:
            tau, n, neighborhood = find_opt_params(file_path)
            taus.append(tau)
            ns.append(n)
            neighborhoods.append(neighborhood)
        except Exception as e:
            print(f"  ERROR processing {file_path.name}: {e}")
            continue
    
    if len(taus) == 0:
        print(f"  WARNING: No valid parameters found for this group!")
        return None
    
    # Aggregate parameters across sampled files:
    # - τ (time delay): median
    # - n (embedding dimension): mode (most frequent integer)
    # - neighborhood: median
    avg_tau = int(np.median(taus))
    avg_n = int(np.bincount(np.array(ns, dtype=int)).argmax())
    avg_neighborhood = float(np.median(neighborhoods))
    
    # Round to reasonable precision
    avg_tau = int(round(avg_tau))
    avg_n = int(round(avg_n))
    
    print(f"\n  Average parameters:")
    print(f"    Time delay (τ): {avg_tau}")
    print(f"    Embedding dimension (n): {avg_n}")
    print(f"    Neighborhood radius: {avg_neighborhood:.6f}")
    
    return {
        'category': category,
        'cop_type': cop_type,
        'affected_side': affected_side,
        'eye': eye_condition,
        'tau': avg_tau,
        'n': avg_n,
        'neighborhood': avg_neighborhood,
        'num_files_processed': len(taus)
    }

def extract_optimal_params(healthy_dir, stroke_dir):
    all_group_params = []

    # Process HEALTHY groups
    # Structure: healthy/COP_type/eye_condition/axis/
    print("\n" + "="*60)
    print("PROCESSING HEALTHY GROUPS")
    print("="*60)

    for cop_dir in healthy_dir.iterdir():
        if not cop_dir.is_dir():
            continue
        cop_type = cop_dir.name  # "COP1", "COP2", or "COP"
        
        for eye_dir in cop_dir.iterdir():
            if not eye_dir.is_dir():
                continue
            eye_condition = eye_dir.name  # "eyes_open" or "eyes_closed"
            
            for axis_dir in eye_dir.iterdir():
                if not axis_dir.is_dir():
                    continue
                axis = axis_dir.name  # "x" or "y"
                
                # Get all files in this group
                group_files = list(axis_dir.glob("*.txt"))
                
                if len(group_files) == 0:
                    print(f"\nSkipping {cop_type}/{eye_condition}/{axis} - no files found")
                    continue
                
                # Find optimal parameters for this group
                group_params = find_group_optimal_params(
                    group_files, 
                    category="healthy",
                    cop_type=cop_type,
                    eye_condition=eye_condition,
                    affected_side="",
                    axis = axis
                )
                
                if group_params:
                    # Add axis to the returned dict
                    group_params['axis'] = axis
                    all_group_params.append(group_params)

    # Process STROKE groups
    # Structure: stroke/affected_side/COP_type/eye_condition/axis/
    print("\n" + "="*60)
    print("PROCESSING STROKE GROUPS")
    print("="*60)

    for affected_dir in stroke_dir.iterdir():
        if not affected_dir.is_dir():
            continue
        affected_side = affected_dir.name  # "left_affected" or "right_affected"
        
        for cop_dir in affected_dir.iterdir():
            if not cop_dir.is_dir():
                continue
            cop_type = cop_dir.name  # "COP1", "COP2", or "COP"
            
            for eye_dir in cop_dir.iterdir():
                if not eye_dir.is_dir():
                    continue
                eye_condition = eye_dir.name  # "eyes_open" or "eyes_closed"
                
                for axis_dir in eye_dir.iterdir():
                    if not axis_dir.is_dir():
                        continue
                    axis = axis_dir.name  # "x" or "y"
                    
                    # Get all files in this group
                    group_files = list(axis_dir.glob("*.txt"))
                    
                    if len(group_files) == 0:
                        print(f"\nSkipping {affected_side}/{cop_type}/{eye_condition}/{axis} - no files found")
                        continue
                    
                    # Find optimal parameters for this group
                    group_params = find_group_optimal_params(
                        group_files,
                        category="stroke",
                        cop_type=cop_type,
                        eye_condition=eye_condition,
                        affected_side=affected_side
                    )
                    
                    if group_params:
                        # Add axis to the returned dict
                        group_params['axis'] = axis
                        all_group_params.append(group_params)
    return all_group_params


def main():
    optimal_params = extract_optimal_params(SORTED_HEALTHY, SORTED_STROKE)
    print("EXTRACTED OPTIMAL PARAMETERS")

    optimal_params_file = EXTRACTED / "optimal_params.csv"
    with open(optimal_params_file, 'w') as f:
        f.write("category,cop_type,affected_side,eye,axis,tau,n,neighborhood,num_files\n")

    print("WRITING RESULTS TO CSV")
    print("="*60)

    with open(optimal_params_file, 'a') as f:
        for params in optimal_params:
            f.write(f"{params['category']},{params['cop_type']},{params['affected_side']},")
            f.write(f"{params['eye']},{params['axis']},{params['tau']},{params['n']},")
            f.write(f"{params['neighborhood']:.6f},{params['num_files_processed']}\n")

    print(f"\n✓ Completed! Found optimal parameters for {len(optimal_params)} groups")
    print(f"✓ Results saved to: {optimal_params_file}")

    # Display summary
    print("\n" + "="*60)
    print("SUMMARY OF OPTIMAL PARAMETERS")
    print("="*60)
    for params in optimal_params:
        print(f"{params['category']:8s} | {params['affected_side']:15s} | {params['cop_type']:5s} | {params['eye']:12s} | {params['axis']:3s} | "
            f"τ={params['tau']:3d} | n={params['n']:2d} | ε={params['neighborhood']:.6f}")
    

#USED FOR TESTING
# if __name__ == "__main__":
#     main()