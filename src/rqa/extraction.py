"""Parameter-selection workflow for group-level RQA settings.

This module estimates optimal embedding parameters for each experimental group
in the sorted COP dataset. Parameters are estimated per file and then
aggregated at the group level for downstream, fixed-parameter RQA analysis.
"""

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

def find_opt_params(data_file):
    """Estimate per-file embedding and neighborhood parameters.

    Parameters
    ----------
    data_file : str or pathlib.Path
        Path to a single COP signal file.

    Returns
    -------
    tuple[int, int, float]
        Estimated ``(tau, n, neighborhood)`` for the input file.

    Notes
    -----
    The signal is z-score normalized before parameter estimation.
    """
    print(f"Processing: {data_file}")

    with open(data_file, 'r') as file:
        data_points = file.readlines()
    data_points = np.array(data_points, dtype=np.float64)

    # Apply per-file z-score normalization before delay/FNN estimation so that
    # neighborhood optimization is less sensitive to subject-specific amplitude.
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

def find_group_optimal_params(group_files, category, cop_type, eye_condition, affected_side="", axis=""):
    """Aggregate optimal parameters for one stratified subgroup.

    Parameters
    ----------
    group_files : list[pathlib.Path]
        File paths belonging to one subgroup.
    category : str
        Subject class label (for example ``"healthy"`` or ``"stroke"``).
    cop_type : str
        COP channel label (for example ``"COP1"``, ``"COP2"``, or ``"COP"``).
    eye_condition : str
        Visual condition label.
    affected_side : str, default ""
        Affected-side label for stroke groups.
    axis : str, default ""
        Signal axis label.

    Returns
    -------
    dict or None
        Dictionary of aggregated parameters for the subgroup, or ``None`` when
        no valid file-level estimates are produced.

    Notes
    -----
    Aggregation uses median for ``tau`` and neighborhood radius, and mode for
    embedding dimension ``n``.
    """
    # Sample up to 30 files to keep computation bounded while preserving
    # representative parameter estimates for each subgroup.
    sample_size = min(30, len(group_files))
    sampled_files = random.sample(group_files, sample_size) if len(group_files) > sample_size else group_files
    
    print(f"\n{'='*60}")
    print(f"Processing group: {category}/{affected_side}/{cop_type}/{eye_condition}/{axis}")
    print(f"Total files: {len(group_files)}, Sampling: {len(sampled_files)}")
    print(f"{'='*60}")
    
    taus = []
    ns = []
    neighborhoods = []
    
    # Estimate parameters file-by-file, then summarize robustly.
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
    
    # Aggregate rules used for publication reproducibility:
    # τ (time delay) -> median, n (embedding dimension) -> mode,
    # neighborhood radius -> median.
    avg_tau = int(np.median(taus))
    avg_n = int(np.bincount(np.array(ns, dtype=int)).argmax())
    avg_neighborhood = float(np.median(neighborhoods))
    
    # Keep integer-valued embedding parameters explicit.
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
    """Estimate optimal RQA parameters for all healthy and stroke subgroups.

    Parameters
    ----------
    healthy_dir : pathlib.Path
        Root folder for healthy subgroup files.
    stroke_dir : pathlib.Path
        Root folder for stroke subgroup files.

    Returns
    -------
    list[dict]
        Aggregated parameter records for all detected subgroups.
    """
    all_group_params = []

    # Process healthy subgroups.
    print("\n" + "="*60)
    print("PROCESSING HEALTHY GROUPS")
    print("="*60)

    for cop_dir in healthy_dir.iterdir():
        if not cop_dir.is_dir():
            continue
        cop_type = cop_dir.name
        
        for eye_dir in cop_dir.iterdir():
            if not eye_dir.is_dir():
                continue
            eye_condition = eye_dir.name
            
            for axis_dir in eye_dir.iterdir():
                if not axis_dir.is_dir():
                    continue
                axis = axis_dir.name
                
                group_files = list(axis_dir.glob("*.txt"))
                
                if len(group_files) == 0:
                    print(f"\nSkipping {cop_type}/{eye_condition}/{axis} - no files found")
                    continue
                
                group_params = find_group_optimal_params(
                    group_files, 
                    category="healthy",
                    cop_type=cop_type,
                    eye_condition=eye_condition,
                    affected_side="",
                    axis = axis
                )
                
                if group_params:
                    group_params['axis'] = axis
                    all_group_params.append(group_params)

    # Process stroke subgroups.
    print("\n" + "="*60)
    print("PROCESSING STROKE GROUPS")
    print("="*60)

    for affected_dir in stroke_dir.iterdir():
        if not affected_dir.is_dir():
            continue
        affected_side = affected_dir.name
        
        for cop_dir in affected_dir.iterdir():
            if not cop_dir.is_dir():
                continue
            cop_type = cop_dir.name
            
            for eye_dir in cop_dir.iterdir():
                if not eye_dir.is_dir():
                    continue
                eye_condition = eye_dir.name
                
                for axis_dir in eye_dir.iterdir():
                    if not axis_dir.is_dir():
                        continue
                    axis = axis_dir.name
                    
                    group_files = list(axis_dir.glob("*.txt"))
                    
                    if len(group_files) == 0:
                        print(f"\nSkipping {affected_side}/{cop_type}/{eye_condition}/{axis} - no files found")
                        continue
                    
                    group_params = find_group_optimal_params(
                        group_files,
                        category="stroke",
                        cop_type=cop_type,
                        eye_condition=eye_condition,
                        affected_side=affected_side
                    )
                    
                    if group_params:
                        group_params['axis'] = axis
                        all_group_params.append(group_params)
    return all_group_params


def main():
    """Run subgroup parameter extraction and write ``optimal_params.csv``.

    Returns
    -------
    None
        This function writes outputs to disk and logs progress to stdout.
    """
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

    print("\n" + "="*60)
    print("SUMMARY OF OPTIMAL PARAMETERS")
    print("="*60)
    for params in optimal_params:
        print(f"{params['category']:8s} | {params['affected_side']:15s} | {params['cop_type']:5s} | {params['eye']:12s} | {params['axis']:3s} | "
            f"τ={params['tau']:3d} | n={params['n']:2d} | ε={params['neighborhood']:.6f}")
    

# NOTE: Local debug entrypoint retained as commented code by design.
# if __name__ == "__main__":
#     main()