"""Core RQA execution helpers for per-file metric/plot generation."""

from pathlib import Path
from unittest import result
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.image_generator import ImageGenerator

from rqa.rqa_utils import determine_group_from_path
from rqa.metrics import calculate_rqa_metrics
from rqa.plotting import generate_rqa_plot


def compute_rqa(file_path, optimal_params_dict, RQA_PLOTS_DIR):
    """Compute RQA metrics and export a recurrence plot for one file.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to one COP signal file.
    optimal_params_dict : dict
        Mapping from subgroup keys to ``(tau, n, neighborhood)`` tuples.
    RQA_PLOTS_DIR : pathlib.Path
        Root directory where recurrence-plot images are written.

    Returns
    -------
    dict or None
        Metrics dictionary with subgroup metadata, or ``None`` when the file
        cannot be processed.
    """
    group = determine_group_from_path(file_path)
    if group is None:
        print(f"  WARNING: Could not determine group for {file_path.name}")
        return None
    
    category, affected_side, cop_type, eye_condition, axis = group
    
    if group not in optimal_params_dict:
        print(f"  WARNING: No optimal parameters found for group {group} - file {file_path.name}")
        return None
    
    tau, n, neighborhood = optimal_params_dict[group]

    rqa_settings = compute_rqa_settings(file_path, tau, n, neighborhood)
    print(type(rqa_settings))
    rqa_metrics = calculate_rqa_metrics(rqa_settings, file_path.name)

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
    plot_path = plot_dir / file_path.name.replace('.txt', '.png')
    
    ImageGenerator.save_recurrence_plot(
        rqa_plot.recurrence_matrix_reverse,
        plot_path
    )

    return rqa_metrics


def compute_rqa_settings(file_path, tau, n, neighborhood):
    """Create a pyrqa ``Settings`` object for one signal.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to one COP signal file.
    tau : int
        Embedding delay.
    n : int
        Embedding dimension.
    neighborhood : float
        Fixed radius used to define recurrent points.

    Returns
    -------
    pyrqa.settings.Settings or None
        Configured settings object, or ``None`` if setup fails.
    """
    try:
        with open(file_path, 'r') as file:
            data_points = file.readlines()
        
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