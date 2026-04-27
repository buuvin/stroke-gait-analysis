from pyrqa.settings import Settings
from pathlib import Path
import shutil

from paths import SORTED_DATA, SORTED_HEALTHY, SORTED_STROKE, EXTRACTED, RAW_PLOTS, SORTED_PLOTS, OPTIMAL_PARAMS_FILE, RQA_METRICS_FILE
from rqa.rqa_core import compute_rqa
from rqa.extraction import extract_optimal_params
from rqa.rqa_utils import load_optimal_params



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
    

#GENERATE RQA PLOTS AND METRICS FOR ALL FILES USING OPTIMAL PARAMETERS

# Load optimal parameters
print("Loading optimal parameters...")
optimal_params_dict = load_optimal_params(OPTIMAL_PARAMS_FILE)
print(f"Loaded optimal parameters for {len(optimal_params_dict)} groups\n")

# Initialize metrics file with header
with open(RQA_METRICS_FILE, 'w') as f:
    f.write("filename,category,affected_side,cop_type,eye_condition,axis,")
    f.write("recurrence_rate,determinism,laminarity,entropy_diagonal_lines,max_diag_line,")
    f.write("max_vert_line,trapping_time,avg_diag_line,avg_vert_line,divergence,det_rr_ratio,lam_det_ratio, det_zero\n")

all_metrics = []
total_files = 0
processed_files = 0

# Process all files in sorted directory structure
print("="*60)
print("PROCESSING ALL FILES")
print("="*60)

# Process HEALTHY files
# Structure: healthy/COP_type/eye_condition/axis/
print("\nProcessing HEALTHY files...")
for cop_dir in (SORTED_HEALTHY).iterdir():
    if not cop_dir.is_dir():
        continue
    
    for eye_dir in cop_dir.iterdir():
        if not eye_dir.is_dir():
            continue
        
        for axis_dir in eye_dir.iterdir():
            if not axis_dir.is_dir():
                continue
            
            files = list(axis_dir.glob("*.txt"))
            total_files += len(files)
            
            for file_path in files:
                metrics = compute_rqa(file_path, optimal_params_dict, RAW_PLOTS)
                if metrics:
                    all_metrics.append(metrics)
                    processed_files += 1
                    if processed_files % 50 == 0:
                        print(f"  Processed {processed_files} files...")

# Process STROKE files
# Structure: stroke/affected_side/COP_type/eye_condition/axis/
print("\nProcessing STROKE files...")
for affected_dir in (SORTED_STROKE).iterdir():
    if not affected_dir.is_dir():
        continue
    
    for cop_dir in affected_dir.iterdir():
        if not cop_dir.is_dir():
            continue
        
        for eye_dir in cop_dir.iterdir():
            if not eye_dir.is_dir():
                continue
            
            for axis_dir in eye_dir.iterdir():
                if not axis_dir.is_dir():
                    continue
                
                files = list(axis_dir.glob("*.txt"))
                total_files += len(files)
                
                for file_path in files:
                    metrics = compute_rqa(file_path, optimal_params_dict, RAW_PLOTS)
                    if metrics:
                        all_metrics.append(metrics)
                        processed_files += 1
                        if processed_files % 50 == 0:
                            print(f"  Processed {processed_files} files...")

# Write all metrics to CSV
print("\n" + "="*60)
print("WRITING METRICS TO CSV")
print("="*60)

with open(RQA_METRICS_FILE, 'a') as f:
    for metrics in all_metrics:
        f.write(f"{metrics['filename']},{metrics['category']},{metrics['affected_side']},")
        f.write(f"{metrics['cop_type']},{metrics['eye_condition']},{metrics['axis']},")
        f.write(f"{metrics['recurrence_rate']:.6f},{metrics['determinism']:.6f},")
        f.write(f"{metrics['laminarity']:.6f},{metrics['entropy']:.6f},")
        f.write(f"{metrics['max_diag_line']},{metrics['max_vert_line']},")
        f.write(f"{metrics['trapping_time']:.6f},{metrics['avg_diag_line']:.6f},")
        f.write(f"{metrics['avg_vert_line']:.6f},")
        f.write(f"{metrics['divergence']:.6f},{metrics['det_rr']:.6f},")
        f.write(f"{metrics['lam_det']:.6f},")
        f.write(f"{metrics['det_zero']:.6f}\n")

print(f"\n✓ Completed!")
print(f"  Total files found: {total_files}")
print(f"  Successfully processed: {processed_files}")
print(f"  Metrics saved to: {RQA_METRICS_FILE}")
print(f"  Plots saved to: {RAW_PLOTS}")

# SORT RQA PLOTS

src_dir = RAW_PLOTS
dst_root = SORTED_PLOTS

for file in src_dir.glob("*.png"):
    name = file.name.upper() 


    if name.startswith("SUP"):
        cat = "healthy"
    elif name.startswith("CK"):
        cat = "stroke"
    else:
        continue


    if "SEC" in name:
        cond = "eyes_closed"
    elif "SEO" in name:
        cond = "eyes_open"
    else:
        continue


    if name.endswith("_X.PNG"):
        side = "left"
    elif name.endswith("_Y.PNG"):
        side = "right"
    else:
        side = "combined"


    dest = dst_root / cat / side / cond
    dest.mkdir(parents=True, exist_ok=True)


    shutil.copy(str(file), dest / file.name)
    print(f"Copied {file.name} → {dest}")