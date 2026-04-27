"""Helpers for generating recurrence plot matrices with pyrqa."""

from pyrqa.computation import RPComputation

def generate_rqa_plot(settings, file_name="plot.png"):
    """Compute a recurrence plot result object for one signal.

    Parameters
    ----------
    settings : pyrqa.settings.Settings
        Configured pyrqa settings object.
    file_name : str, default "plot.png"
        File label used for error context in logs.

    Returns
    -------
    object or None
        pyrqa recurrence-plot computation result on success, else ``None``.
    """
    try:
        computation = RPComputation.create(settings, verbose=False)
        result = computation.run()
        return result
    except Exception as e:
        print(f"  ERROR generating RQA plot for {file_name}: {e}")

