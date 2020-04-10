from tools.analyze import *
from parameters import parameters

if __name__ == "__main__":
    """Run experiments according to the parameters from parameters.py and analyze the results."""

    # run the experiments and save the data
    data = run_in_parallel(parameters)
    file_path = save_experiment_results(parameters, data)

    # analyze the results
    my_analyzer = Analyzer(file_path, save_figures=True)
    my_analyzer.display_mapping()
    my_analyzer.display_mapping_evolvability()
    my_analyzer.display_all_stats()
    my_analyzer.display_run_stats()
    my_analyzer.display_search_history()
    my_analyzer.display_selected_individuals()
    my_analyzer.display_novelty(generation=0)

    plt.show(block=True)
