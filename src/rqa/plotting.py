from pyrqa.computation import RPComputation

def generate_rqa_plot(settings, file_name="plot.png"):
    try:
        computation = RPComputation.create(settings, verbose=False)
        result = computation.run()
        return result
    except Exception as e:
        print(f"  ERROR generating RQA plot for {file_name}: {e}")

