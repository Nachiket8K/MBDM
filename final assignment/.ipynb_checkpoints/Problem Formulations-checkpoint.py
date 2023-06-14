# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# make sure pandas is version 1.0 or higher
# make sure networkx is verion 2.4 or higher
print(pd.__version__)
print(nx.__version__)

# +
from ema_workbench import (Model, Policy, ema_logging, SequentialEvaluator, 
                           MultiprocessingEvaluator)
from dike_model_function import DikeNetwork  # @UnresolvedImport
from problem_formulation import get_model_for_problem_formulation

def sum_over(*args):
    return sum(args)


# +
ema_logging.log_to_stderr(ema_logging.INFO)

#choose problem formulation number, between 0-5
#each problem formulation has its own list of outcomes
dike_model, planning_steps = get_model_for_problem_formulation(3)

# +
#enlisting uncertainties, their types (RealParameter/IntegerParameter/CategoricalParameter), lower boundary, and upper boundary
import copy
for unc in dike_model.uncertainties:
    print(repr(unc))
    
uncertainties = copy.deepcopy(dike_model.uncertainties)

# +
#enlisting policy levers, their types (RealParameter/IntegerParameter), lower boundary, and upper boundary
for policy in dike_model.levers:
    print(repr(policy))
    
levers = copy.deepcopy(dike_model.levers)
# -

#enlisting outcomes
for outcome in dike_model.outcomes:
    print(repr(outcome))

#running the model through EMA workbench 
with MultiprocessingEvaluator(dike_model) as evaluator:
    results = evaluator.perform_experiments(scenarios=50, policies=4)

#observing the simulation runs
experiments, outcomes = results
print(outcomes.keys())
experiments

#

# only works because we have scalar outcomes
pd.DataFrame(outcomes)


# +
# defining specific policies
# for example, policy 1 is about extra protection in upper boundary
# policy 2 is about extra protection in lower boundary
# policy 3 is extra protection in random locations

def get_do_nothing_dict():
    return {l.name:0 for l in dike_model.levers}


policies = [Policy('policy 1', **dict(get_do_nothing_dict(), 
                                    **{'0_RfR 0':1,
                                      '0_RfR 1':1,
                                      '0_RfR 2':1,
                                      'A.1_DikeIncrease 0':5})),
           Policy('policy 2', **dict(get_do_nothing_dict(), 
                                   **{'4_RfR 0':1,
                                      '4_RfR 1':1,
                                      '4_RfR 2':1,
                                      'A.5_DikeIncrease 0':5})),
           Policy('policy 3', **dict(get_do_nothing_dict(),
                                   **{'1_RfR 0':1,
                                      '2_RfR 1':1,
                                      '3_RfR 2':1,
                                      'A.3_DikeIncrease 0':5}))]
# -

#pass the policies list to EMA workbench experiment runs
n_scenarios = 100
with MultiprocessingEvaluator(dike_model) as evaluator:
    results = evaluator.perform_experiments(n_scenarios,
                                            policies)



# only works because we have scalar outcomes
pd.DataFrame(results[1])


