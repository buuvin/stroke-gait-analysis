from pathlib import Path
from unittest import result
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.image_generator import ImageGenerator

from src.rqa.rqa_utils import determine_group_from_path
from src.rqa.metrics import extract_rqa_metrics
from src.rqa.plotting import generate_rqa_plot


def compute_rqa(file_path, optimal_params_dict, RQA_PLOTS_DIR):
    """
    Compute RQA metrics and generate plot for a single file using group-specific optimal parameters.
    
    Args:
        file_path: Path to the data file
        optimal_params_dict: Dictionary of optimal parameters by group
        RQA_PLOTS_DIR: Directory to save plots
    """
    # Determine group from file path
    group = determine_group_from_path(file_path)
    if group is None:
        print(f"  WARNING: Could not determine group for {file_path.name}")
        return None
    
    category, affected_side, cop_type, eye_condition, axis = group
    
    # Look up optimal parameters
    if group not in optimal_params_dict:
        print(f"  WARNING: No optimal parameters found for group {group} - file {file_path.name}")
        return None
    
    tau, n, neighborhood = optimal_params_dict[group]

    rqa_settings = compute_rqa_settings(file_path, tau, n, neighborhood)

    rqa_metrics = extract_rqa_metrics(rqa_settings, file_path.name)

     # Add group information
    rqa_metrics['category'] = category
    rqa_metrics['affected_side'] = affected_side
    rqa_metrics['cop_type'] = cop_type
    rqa_metrics['eye_condition'] = eye_condition
    rqa_metrics['axis'] = axis

    rqa_plot = generate_rqa_plot(rqa_settings)
    
    if category == 'healthy':
        plot_dir = RQA_PLOTS_DIR / category / cop_type / eye_condition / axis
    else:
        plot_dir = RQA_PLOTS_DIR / category / affected_side / cop_type / eye_condition / axis
    
    plot_dir.mkdir(parents=True, exist_ok=True)
    # Preserve original filename: e.g., "CK09_PSEC3_COP_X.txt" -> "CK09_PSEC3_COP_X.png"
    plot_path = plot_dir / file_path.name.replace('.txt', '.png')
    
    ImageGenerator.save_recurrence_plot(
        rqa_plot.recurrence_matrix_reverse,
        plot_path
    )

    return rqa_metrics


def compute_rqa_settings(file_path, tau, n, neighborhood):
    """
    Compute RQA metrics and generate plot for a single file using group-specific optimal parameters.
    
    Args:
        file_path: Path to the data file
        optimal_params_dict: Dictionary of optimal parameters by group
        RQA_PLOTS_DIR: Directory to save plots
    
    Returns:
        dict: RQA metrics with file information, or None if error
    """
    try:
        # Load data
        with open(file_path, 'r') as file:
            data_points = file.readlines()
        
        # Create time series and settings with optimal parameters
        time_series = TimeSeries(data_points,
                                embedding_dimension=n,
                                time_delay=tau)
        settings = Settings(time_series,
                            analysis_type=Classic,
                            neighbourhood=FixedRadius(neighborhood),
                            similarity_measure=EuclideanMetric,
                            theiler_corrector=tau)
        return settings
    
    except Exception as e:
        print(f"  ERROR building settings for {Path(file_path).name}: {e}")
        return None