import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

print(pd.__version__)
print(nx.__version__)

# Rest of the code goes here...

if __name__ == '__main__':
    from ema_workbench import (Model, MultiprocessingEvaluator, Policy, Scenario)
    from ema_workbench.em_framework.evaluators import perform_experiments
    from ema_workbench.em_framework.samplers import sample_uncertainties
    from ema_workbench.util import ema_logging
    import time
    from problem_formulation import get_model_for_problem_formulation

    ema_logging.log_to_stderr(ema_logging.INFO)

    # choose problem formulation number, between 0-5
    # each problem formulation has its own list of outcomes
    dike_model, planning_steps = get_model_for_problem_formulation(3)

    uncertainties = dike_model.uncertainties
    levers = dike_model.levers

    # Rest of the code goes here...

    def main():
        ema_logging.log_to_stderr(ema_logging.INFO)

        # Make use of the multiprocessing evaluator to save time
        # running with 2000 scenarios
        with MultiprocessingEvaluator(dike_model, n_processes=7) as evaluator:
            results = evaluator.perform_experiments(scenarios=4000, policies=1)

        # Save results for later analysis
        from ema_workbench import save_results
        save_results(results, 'results/4000 scenarios 1 policy PF1.tar.gz')

    if __name__ == '__main__':
        from multiprocessing import freeze_support
        freeze_support()
        main()
