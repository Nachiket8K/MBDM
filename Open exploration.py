# General libraries
import copy

from multiprocessing import freeze_support

# EMA workbench imports
from ema_workbench import (Policy, MultiprocessingEvaluator)
from ema_workbench.util import ema_logging, save_results

# Import problem formulation
from problem_formulation import get_model_for_problem_formulation

ema_logging.log_to_stderr(ema_logging.INFO)
# define the problem formulation between 0 and 9
# Problem formulation 6 is specific to Dike Ring 3
dike_model, planning_steps = get_model_for_problem_formulation(6)

# Set uncertainties and levers variables
uncertainties = copy.deepcopy(dike_model.uncertainties)
levers = copy.deepcopy(dike_model.levers)

# Create a reference policy where no action is taken
nullPolicy = [Policy('null_policy', **{'0_RfR 0': 0,
                                       '0_RfR 1': 0,
                                       '0_RfR 2': 0,
                                       'A.1_DikeIncrease 0': 0,
                                       'A.2_DikeIncrease 0': 0,
                                       'A.3_DikeIncrease 0': 0,
                                       'A.4_DikeIncrease 0': 0,
                                       'A.5_DikeIncrease 0': 0,
                                       'A.1_DikeIncrease 1': 0,
                                       'A.2_DikeIncrease 1': 0,
                                       'A.3_DikeIncrease 1': 0,
                                       'A.4_DikeIncrease 1': 0,
                                       'A.5_DikeIncrease 1': 0,
                                       'A.1_DikeIncrease 2': 0,
                                       'A.2_DikeIncrease 2': 0,
                                       'A.3_DikeIncrease 2': 0,
                                       'A.4_DikeIncrease 2': 0,
                                       'A.5_DikeIncrease 2': 0,
                                       'EWS_DaysToThreat': 0})
              ]
def main():
    n_scenarios = 100

    ema_logging.log_to_stderr(ema_logging.INFO)

    with MultiprocessingEvaluator(dike_model) as evaluator:
        experiments, outcomes = evaluator.perform_experiments(scenarios=n_scenarios, policies=nullPolicy)

    # Save the results
    save_results([experiments, outcomes], 'results/100Scenarios_NullPolicy_PF6.tar.gz')

# Create a policy where action is taken randomly
# compare the performance of that policy to the previously computed null_policy

'''
import random  # for randomly deciding to switch policy on/ off

alphaPolicy = [Policy('alpha_random_policy', **{'3_RfR 0': random.randint(0, 1),
                                                '3_RfR 1': random.randint(0, 1),
                                                '3_RfR 2': random.randint(0, 1),
                                                'A.1_DikeIncrease 0': random.randint(0, 10),
                                                'A.2_DikeIncrease 0': random.randint(0, 10),
                                                'A.3_DikeIncrease 0': random.randint(0, 10),
                                                'A.4_DikeIncrease 0': random.randint(0, 10),
                                                'A.5_DikeIncrease 0': random.randint(0, 10),
                                                'A.1_DikeIncrease 1': random.randint(0, 10),
                                                'A.2_DikeIncrease 1': random.randint(0, 10),
                                                'A.3_DikeIncrease 1': random.randint(0, 10),
                                                'A.4_DikeIncrease 1': random.randint(0, 10),
                                                'A.5_DikeIncrease 1': random.randint(0, 10),
                                                'A.1_DikeIncrease 2': random.randint(0, 10),
                                                'A.2_DikeIncrease 2': random.randint(0, 10),
                                                'A.3_DikeIncrease 2': random.randint(0, 10),
                                                'A.4_DikeIncrease 2': random.randint(0, 10),
                                                'A.5_DikeIncrease 2': random.randint(0, 10),
                                                'EWS_DaysToThreat': random.randint(0, 15)})]

betaPolicy = [Policy('beta_random_policy', **{'3_RfR 0': random.randint(0, 1),
                                              '3_RfR 1': random.randint(0, 1),
                                              '3_RfR 2': random.randint(0, 1),
                                              'A.1_DikeIncrease 0': random.randint(0, 10),
                                              'A.2_DikeIncrease 0': random.randint(0, 10),
                                              'A.3_DikeIncrease 0': random.randint(0, 10),
                                              'A.4_DikeIncrease 0': random.randint(0, 10),
                                              'A.5_DikeIncrease 0': random.randint(0, 10),
                                              'A.1_DikeIncrease 1': random.randint(0, 10),
                                              'A.2_DikeIncrease 1': random.randint(0, 10),
                                              'A.3_DikeIncrease 1': random.randint(0, 10),
                                              'A.4_DikeIncrease 1': random.randint(0, 10),
                                              'A.5_DikeIncrease 1': random.randint(0, 10),
                                              'A.1_DikeIncrease 2': random.randint(0, 10),
                                              'A.2_DikeIncrease 2': random.randint(0, 10),
                                              'A.3_DikeIncrease 2': random.randint(0, 10),
                                              'A.4_DikeIncrease 2': random.randint(0, 10),
                                              'A.5_DikeIncrease 2': random.randint(0, 10),
                                              'EWS_DaysToThreat': random.randint(0, 15)})]
gammaPolicy = [Policy('gamma_random_policy', **{'3_RfR 0': random.randint(0, 1),
                                                '3_RfR 1': random.randint(0, 1),
                                                '3_RfR 2': random.randint(0, 1),
                                                'A.1_DikeIncrease 0': random.randint(0, 10),
                                                'A.2_DikeIncrease 0': random.randint(0, 10),
                                                'A.3_DikeIncrease 0': random.randint(0, 10),
                                                'A.4_DikeIncrease 0': random.randint(0, 10),
                                                'A.5_DikeIncrease 0': random.randint(0, 10),
                                                'A.1_DikeIncrease 1': random.randint(0, 10),
                                                'A.2_DikeIncrease 1': random.randint(0, 10),
                                                'A.3_DikeIncrease 1': random.randint(0, 10),
                                                'A.4_DikeIncrease 1': random.randint(0, 10),
                                                'A.5_DikeIncrease 1': random.randint(0, 10),
                                                'A.1_DikeIncrease 2': random.randint(0, 10),
                                                'A.2_DikeIncrease 2': random.randint(0, 10),
                                                'A.3_DikeIncrease 2': random.randint(0, 10),
                                                'A.4_DikeIncrease 2': random.randint(0, 10),
                                                'A.5_DikeIncrease 2': random.randint(0, 10),
                                                'EWS_DaysToThreat': random.randint(0, 15)})]
def main():
    n_scenarios = 5

    ema_logging.log_to_stderr(ema_logging.INFO)

    # Model Imports
    from problem_formulation import get_model_for_problem_formulation

    dike_model, time_step = get_model_for_problem_formulation(7)

    with MultiprocessingEvaluator(dike_model,n_processes=2) as evaluator:
        experiments, outcomes = evaluator.perform_experiments(scenarios=n_scenarios, policies=[nullPolicy,
                                                                                               alphaPolicy,
                                                                                               betaPolicy,
                                                                                               gammaPolicy])

    # Save the results
    save_results([experiments, outcomes], 'results/10000Scenarios_RandomPolicies_PF7.tar.gz')
'''
if __name__ == '__main__':
    freeze_support()
    main()
