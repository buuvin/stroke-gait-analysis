from pyrqa.computation import RQAComputation

def extract_rqa_metrics(result, filename = ""):
    return {
        "filename" : filename,
        "recurrence_rate": result.recurrence_rate,
        "determinism": result.determinism,
        "divergence": result.divergence,
        "det_rr": result.determinism / result.recurrence_rate,
        "laminarity": result.laminarity,
        "lam_det": result.laminarity / (result.determinism + 1e-12),
        "entropy": result.entropy_diagonal_lines,
        "max_diag_line": result.longest_diagonal_line,
        "max_vert_line": result.longest_vertical_line,
        "trapping_time": result.trapping_time,
        "avg_diag_line": result.average_diagonal_line,
        "avg_vert_line": result.average_white_vertical_line,
        "det_zero": int(result.determinism < 1e-6)
    }

def calculate_rqa_metrics(settings, file_name):
    try:
        computation = RQAComputation.create(settings, verbose=False)
        result = computation.run()
        return extract_rqa_metrics(result, filename=file_name)
    except Exception as e:
        print(f"  ERROR computing RQA for {file_name}: {e}")
        return None