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

# # Open exploration
# This Notebook focusses on Scenario Discovery, and it devised to be used before the directed search notebook has run. Several different methods are used to do this scenario discovery, including PRIM, dimensional stacking, global sensitivity analysis, and feature scoring. In most cases of this analysis, two problem formulations are used - Problem formulation 1 (provides aggregated outcomes for the whole IJssel river area), and Problem formulation 2 (provides outcomes for each dike ring which are later then aggregated over only Gelderland dike rings).
#
# In an attempt to save time and keep this notebook efficient, results of past simulations were saved as compressed files, and then only loaded and visualised here. Results were also generated using this notebook, but on a one-at-a-time basis.
#
# ## Import of libraries & Definition of model input and model
#
# The required libraries to run this notebook are:
#
# * ema_workbench
# * seaborn
# * numpy
# * pandas
# * matplotlib.pyplot
# * __future__
# * problem_formulation (.py-file of the model)
# * copy
#
# ## Table of contents:
#     1. Scenario Discovery
#         a. Sample generation over uncertainty space
#         b. Aggregation of outputs (only for problem formulation 3)
#         c. Analysis using PRIM
#     2. Global Sensitivity Analysis
#         a. Sample generation over uncertainty and lever spaces respectively, using SOBOL
#         b. Feature scoring

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ema_workbench import (Model, MultiprocessingEvaluator, Policy, Scenario)

from ema_workbench.em_framework.evaluators import perform_experiments
from ema_workbench.em_framework.samplers import sample_uncertainties
from ema_workbench.util import ema_logging
from problem_formulation import get_model_for_problem_formulation
import copy


ema_logging.log_to_stderr(ema_logging.INFO)
# -

#choose problem formulation number, between 0-5
#each problem formulation has its own list of outcomes
dike_model, planning_steps = get_model_for_problem_formulation(1)

# Set uncertainties and levers variables
uncertainties = copy.deepcopy(dike_model.uncertainties)
levers = copy.deepcopy(dike_model.levers)

# # 1. Scenario Discovery
# ## a. Generate random scenarios over uncertainty space

# +
#running the model through EMA workbench
from ema_workbench import (MultiprocessingEvaluator, ema_logging,
                           perform_experiments, SequentialEvaluator)
ema_logging.log_to_stderr(ema_logging.INFO)
 
# Make use of the multiprocessing evaluator to save time
# running with 2000 scenarios
with MultiprocessingEvaluator(dike_model, n_processes=7) as evaluator:
    results = evaluator.perform_experiments(scenarios=4000, policies=1)

# Save results for later analysis
from ema_workbench import save_results
save_results(results, 'results/4000 scenarios 1 policy PF1.tar.gz')
# -

# ### Problem Formulation 1

#choose problem formulation number, between 0-5
#each problem formulation has its own list of outcomes
dike_model, planning_steps = get_model_for_problem_formulation(1)

# +
from ema_workbench import load_results

# Load results that were previously generated
load_file_name ='results/4000 scenarios 1 policy PF1.tar.gz'
results = load_results(load_file_name)

# See what the results look like and what outcomes were captured
experiments, outcomes = results
print(outcomes.keys())

# Create temporary dataframe containing experiments and outcomes
temp = experiments.copy(deep=True)
# outcomes
temp.head()

# +
experiments = temp.copy(deep=True)

# Drop the lever columns from the dataframe to make sure prim only focusses on uncertainties
experiments.drop(experiments.columns[19:], axis=1, inplace=True)

experiments

# -

# ## c. Analysis with PRIM

# +
# Prepare to do prim
from ema_workbench.analysis import prim

# Set the output to the category that we are interested in for this run
outcome_entry = outcomes["Expected Number of Deaths"]

x = experiments
# Set our output threshold to be the 90th percentile of all of those outputs
y_limit = np.quantile(outcome_entry, 0.9)

# Only select outputs that are above this threshold
y = np.array([value > y_limit for value in outcome_entry])

# Conduct our prim analysis using a threshold of 0.8
prim_alg = prim.Prim(x, y, threshold=0.8)

# Get the box that prim found for us
box1 = prim_alg.find_box()
# -

# Show the peeling graph for this prim analysis
box1.show_tradeoff()
plt.show()

# Show scatter plots for this prim analysis
box1.show_pairs_scatter()
plt.show()


# +
# Do dimensional stacking of results
from ema_workbench.analysis import dimensional_stacking

dimensional_stacking.create_pivot_plot(x, y, 2, nbins=5)
plt.show()
# -

# ## b. Aggregate Gelderland dike rings together for Problem Formulation 3

# +
#choose problem formulation number, between 0-5
#each problem formulation has its own list of outcomes
dike_model, planning_steps = get_model_for_problem_formulation(3)

from ema_workbench import load_results

# Load results that were previously generated
load_file_name ='results/4000 scenarios 1 policy PF3.tar.gz'
results = load_results(load_file_name)

# See what the results look like and what outcomes were captured
experiments, outcomes = results
print(outcomes.keys())

# Create temporary dataframe containing experiments and outcomes
temp = experiments.copy(deep=True)
# outcomes
temp.head()

# +
# Aggregate costs and deaths for only the dike rings in Gelderland based on problem formulation 3.


if (load_file_name == 'results/4000 scenarios 1 policy PF3.tar.gz' or load_file_name == "results/1 scenarios 256 policy PF3 SA lever.tar.gz" ):
    n_scen = len(outcomes['A.1 Total Costs'])
    # Dictionary used to hold our newly aggregated outcomes
    agg_outcomes = {}
    # Aggregate all the costs for the three Gelderland dike rings
    arr_costs = [0]*n_scen
    # Aggregate all the deaths for the three Gelderland dike rings
    arr_deaths = [0]*n_scen

    # Go through all the generated scenarios' outcomes and aggregate over the three dike rings of importance
    for i in range(n_scen):
        arr_costs[i] = outcomes['A.1 Total Costs'][i] + outcomes['A.2 Total Costs'][i] + outcomes['A.3 Total Costs'][i]
        arr_deaths[i] = outcomes['A.1_Expected Number of Deaths'][i] + outcomes['A.2_Expected Number of Deaths'][i] + outcomes['A.3_Expected Number of Deaths'][i]

    agg_outcomes['Expected Annual Damage'] = arr_costs
    agg_outcomes['Expected Number of Deaths'] = arr_deaths


# +
experiments = temp.copy(deep=True)

# Drop the lever columns from the dataframe to make sure prim only focusses on uncertainties
experiments.drop(experiments.columns[19:], axis=1, inplace=True)

experiments

# -

# ## c. Analysis with PRIM

# +
# Prepare to do prim
from ema_workbench.analysis import prim

# Set the output to the category that we are interested in for this run
outcome_entry = agg_outcomes["Expected Annual Damage"]

x = experiments
# Set our output threshold to be the 90th percentile of all of those outputs
y_limit = np.quantile(outcome_entry, 0.9)

# Only select outputs that are above this threshold
y = np.array([value > y_limit for value in outcome_entry])

# Conduct our prim analysis using a threshold of 0.8
prim_alg = prim.Prim(x, y, threshold=0.8)

# Get the box that prim found for us
box1 = prim_alg.find_box()

# +
# Prepare to export the box and its scenarios and outcomes for later use in robustness analysis
from ema_workbench import save_results
box_scenarios = experiments.iloc[box1.yi]

box_outcomes = {k:v[box1.yi] for k,v in outcomes.items()}

save_results((box_scenarios, box_outcomes), 'results/scens_expected_deaths.tar.gz')
# -

# Show the peeling graph for this prim analysis
box1.show_tradeoff()
plt.show()

# Show scatter plots for this prim analysis
box1.show_pairs_scatter()
plt.show()


# +
# Do dimensional stacking of results
from ema_workbench.analysis import dimensional_stacking

dimensional_stacking.create_pivot_plot(x, y, 2, nbins=5)
plt.show()
# -

# # 2. Global Sensitivity Analysis
# ## a. Sample generation over uncertainty and lever spaces respectively, using SOBOL

# +
from SALib.analyze import sobol
from ema_workbench import Samplers
from ema_workbench.em_framework.salib_samplers import get_SALib_problem
#running the model through EMA workbench
from ema_workbench import (MultiprocessingEvaluator, ema_logging,
                           perform_experiments, SequentialEvaluator)
from problem_formulation import get_model_for_problem_formulation
# from ema_workbench.em_framework.evaluators import SOBOL
from ema_workbench import Policy, Scenario

# Create a null policy where no action is taken
policies = [Policy('policy 1', **{'0_RfR 0':0,
                                  '0_RfR 1':0,
                                  '0_RfR 2':0,
                                  'A.1_DikeIncrease 0':0,
                                  'A.2_DikeIncrease 0':0,
                                  'A.3_DikeIncrease 0':0,
                                  'A.4_DikeIncrease 0':0,
                                  'A.5_DikeIncrease 0':0,
                                  'A.1_DikeIncrease 1':0,
                                  'A.2_DikeIncrease 1':0,
                                  'A.3_DikeIncrease 1':0,
                                  'A.4_DikeIncrease 1':0,
                                  'A.5_DikeIncrease 1':0,
                                  'A.1_DikeIncrease 2':0,
                                  'A.2_DikeIncrease 2':0,
                                  'A.3_DikeIncrease 2':0,
                                  'A.4_DikeIncrease 2':0,
                                  'A.5_DikeIncrease 2':0,
                                  'EWS_DaysToThreat':0})
                                  ]

# Define a reference scenario with each uncertainty containing the mean value of its range
scenarios = [Scenario('scenario 1', **{'discount rate 0':3,
                                        'discount rate 1':3,
                                        'discount rate 2':3,
                                        'A.1_Bmax':190,
                                        'A.1_Brate':1.5,
                                        'A.1_pfail':0.5,
                                        'A.2_Bmax':190,
                                        'A.2_Brate':1.5,
                                        'A.2_pfail':0.5,
                                        'A.3_Bmax':190,
                                        'A.3_Brate':1.5,
                                        'A.3_pfail':0.5,
                                        'A.4_Bmax':190,
                                        'A.4_Brate':1.5,
                                        'A.4_pfail':0.5,
                                        'A.5_Bmax':190,
                                        'A.5_Brate':1.5,
                                        'A.5_pfail':0.5,
                                        'A.0_ID flood wave shape':66})
                                    ]

ema_logging.log_to_stderr(ema_logging.INFO)

# Use problem formulation 3 (disaggregated dike rings)
dike_model, planning_steps = get_model_for_problem_formulation(3)


# Perform experiments using SOBOL sampling, in this case used for the levers.
with MultiprocessingEvaluator(dike_model) as evaluator:
    sa_results = evaluator.perform_experiments(
        scenarios=scenarios, policies=256, lever_sampling=Samplers.SOBOL
    )

# Save results from the global sensitivity analysis
from ema_workbench import save_results
save_results(sa_results, 'results/1 scenarios 256 policy PF3 SA lever.tar.gz')
# -

# Load results from the global sensitivity analysis
from ema_workbench import load_results
load_file_name = 'results/1 scenarios 256 policy PF3 SA lever.tar.gz'
sa_results = load_results(load_file_name)

print(outcomes.keys())

# +
from ema_workbench.em_framework.salib_samplers import get_SALib_problem
# Convert our problem to SALib format using EMA's builtin function
experiments, outcomes = sa_results
problem = get_SALib_problem(dike_model.levers)

# Using the outputs and samples generated with SOBOL, do the actual global sensitivity analysis (based on variations)
Si = sobol.analyze(
    problem, outcomes["A.1_Expected Number of Deaths"], calc_second_order=True, print_to_console=False
)

# +
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Plot distributions of ST and ST1 of the values
scores_filtered = {k: Si[k] for k in ["ST", "ST_conf", "S1", "S1_conf"]}
Si_df = pd.DataFrame(scores_filtered, index=problem["names"])

sns.set_style("white")
fig, ax = plt.subplots(1)

indices = Si_df[["S1", "ST"]]
err = Si_df[["S1_conf", "ST_conf"]]

indices.plot.bar(yerr=err.values.T, ax=ax)
fig.set_size_inches(8, 6)
fig.subplots_adjust(bottom=0.3)
plt.show()
# -

# ## b. Feature scoring

# +
from ema_workbench.analysis import feature_scoring
from ema_workbench import load_results

plt.rcParams["figure.figsize"] = (8,7.5)

# load_file_name = '2000 scenarios 1 policy PF3.tar.gz'
load_file_name ='results/1 scenarios 256 policy PF3 SA lever.tar.gz'
results = load_results(load_file_name)

experiments, outcomes = results

# Drop levers for other time steps to make graph more compact and readable
experiments.drop(experiments.columns[:19], axis=1, inplace=True)
experiments.drop(columns=["0_RfR 1", "0_RfR 2", "1_RfR 1", "1_RfR 2", "2_RfR 1", "2_RfR 2", "3_RfR 1", "3_RfR 2",
                            "4_RfR 1", "4_RfR 2", "A.1_DikeIncrease 1", "A.1_DikeIncrease 2", "A.2_DikeIncrease 1", "A.2_DikeIncrease 2",
                            "A.3_DikeIncrease 1", "A.3_DikeIncrease 2", "A.4_DikeIncrease 1", "A.4_DikeIncrease 2", "A.5_DikeIncrease 1",
                            "A.5_DikeIncrease 2", "policy"], inplace=True)

x = experiments
y = outcomes

# Get feature scores for the SOBOL sampled inputs and their corresponding outputs
fs = feature_scoring.get_feature_scores_all(x, y)
sns.heatmap(fs, cmap="magma", annot=True)
plt.show()
