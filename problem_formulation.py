"""
Created on Mon Jun 12 2023

@author: Nachiket
"""
from ema_workbench import (
    Model,
    CategoricalParameter,
    ArrayOutcome,
    ScalarOutcome,
    IntegerParameter,
    RealParameter,
)
from dike_model_function import DikeNetwork  # @UnresolvedImport

import numpy as np


def sum_over(*args):
    numbers = []
    for entry in args:
        if entry is not None:  # Skip None values
            try:
                value = sum(entry)
            except TypeError:
                value = entry
            numbers.append(value)
    return sum(numbers)


def sum_over_time(*args):
    data = np.asarray(args)
    summed = data.sum(axis=0)
    return summed


def get_model_for_problem_formulation(problem_formulation_id):
    """Convenience function to prepare DikeNetwork in a way it can be input in the EMA-workbench.
    Specify uncertainties, levers, and outcomes of interest.

    Parameters
    ----------
    problem_formulation_id : int {0, ..., 5}
                             problem formulations differ with respect to the objectives
                             0: Total cost, and casualties
                             1: Expected damages, costs, and casualties
                             2: expected damages, dike investment costs, rfr costs, evacuation cost, and casualties
                             3: costs and casualties disaggregated over dike rings, and room for the river and
                              evacuation costs
                             4: Expected damages, dike investment cost and casualties disaggregated over dike rings
                             and room for the river and evacuation costs
                             5: disaggregate over time and space

    Notes
    -----
    problem formulations 4 and 5 rely on ArrayOutcomes and thus cannot straightforwardly
    be used in optimizations

    """
    # Load the model:
    function = DikeNetwork()
    # workbench model:
    dike_model = Model("dikesnet", function=function)

    # Uncertainties and Levers:
    # Specify uncertainties range:
    Real_uncert = {"Bmax": [30, 350], "pfail": [0, 1]}  # m and [.]
    # breach growth rate [m/day]
    cat_uncert_loc = {"Brate": (1.0, 1.5, 10)}

    cat_uncert = {
        f"discount rate {n}": (1.5, 2.5, 3.5, 4.5) for n in function.planning_steps
    }

    Int_uncert = {"A.0_ID flood wave shape": [0, 132]}
    # Range of dike heightening:
    dike_lev = {"DikeIncrease": [0, 10]}  # dm

    # Series of five Room for the River projects:
    rfr_lev = [f"{project_id}_RfR" for project_id in range(0, 5)]

    # Time of warning: 0, 1, 2, 3, 4 days ahead from the flood
    EWS_lev = {"EWS_DaysToThreat": [0, 4]}  # days

    uncertainties = []
    levers = []

    for uncert_name in cat_uncert.keys():
        categories = cat_uncert[uncert_name]
        uncertainties.append(CategoricalParameter(uncert_name, categories))

    for uncert_name in Int_uncert.keys():
        uncertainties.append(
            IntegerParameter(
                uncert_name, Int_uncert[uncert_name][0], Int_uncert[uncert_name][1]
            )
        )

    # RfR levers can be either 0 (not implemented) or 1 (implemented)
    for lev_name in rfr_lev:
        for n in function.planning_steps:
            lev_name_ = f"{lev_name} {n}"
            levers.append(IntegerParameter(lev_name_, 0, 1))

    # Early Warning System lever
    for lev_name in EWS_lev.keys():
        levers.append(
            IntegerParameter(lev_name, EWS_lev[lev_name][0], EWS_lev[lev_name][1])
        )

    for dike in function.dikelist:
        # uncertainties in the form: locationName_uncertaintyName
        for uncert_name in Real_uncert.keys():
            name = f"{dike}_{uncert_name}"
            lower, upper = Real_uncert[uncert_name]
            uncertainties.append(RealParameter(name, lower, upper))

        for uncert_name in cat_uncert_loc.keys():
            name = f"{dike}_{uncert_name}"
            categories = cat_uncert_loc[uncert_name]
            uncertainties.append(CategoricalParameter(name, categories))

        # location-related levers in the form: locationName_leversName
        for lev_name in dike_lev.keys():
            for n in function.planning_steps:
                name = f"{dike}_{lev_name} {n}"
                levers.append(
                    IntegerParameter(name, dike_lev[lev_name][0], dike_lev[lev_name][1])
                )

    # load uncertainties and levers in dike_model:
    dike_model.uncertainties = uncertainties
    dike_model.levers = levers

    # Problem formulations:
    # Outcomes are all costs, thus they have to minimized:
    direction = ScalarOutcome.MINIMIZE

    # 2-objective PF:
    if problem_formulation_id == 0:
        cost_variables = []
        casualty_variables = []

        for n in function.planning_steps:
            cost_variables.extend(
                [
                    f"{dike}_{e}_{n}"
                    for e in ["Expected Annual Damage", "Dike Investment Costs"]
                    for dike in function.dikelist
                ]
            )
            casualty_variables.extend(
                [
                    f"{dike}_{e}_{n}"
                    for e in ["Expected Number of Deaths"]
                    for dike in function.dikelist
                ]
            )
            cost_variables.append(f"RfR Total Costs_{n}")
            cost_variables.append(f"Expected Evacuation Costs_{n}")

        dike_model.outcomes = [
            ScalarOutcome(
                "All Costs",
                variable_name=[var for var in cost_variables],
                function=sum_over,
                kind=direction,
            ),
            ScalarOutcome(
                "Expected Number of Deaths",
                variable_name=[var for var in casualty_variables],
                function=sum_over,
                kind=direction,
            ),
        ]

    elif problem_formulation_id == 1:
        damage_variables = []
        cost_variables = []
        casualty_variables = []

        for n in function.planning_steps:
            damage_variables.extend(
                [f"{dike}_Expected Annual Damage_{n}" for dike in function.dikelist]
            )
            cost_variables.extend(
                [f"{dike}_Dike Investment Costs_{n}" for dike in function.dikelist]
                + [f"RfR Total Costs_{n}"]
                + [f"Expected Evacuation Costs_{n}"]
            )
            casualty_variables.extend(
                [f"{dike}_Expected Number of Deaths_{n}" for dike in function.dikelist]
            )

        dike_model.outcomes = [
            ScalarOutcome(
                "Expected Annual Damage",
                variable_name=[var for var in damage_variables],
                function=sum_over,
                kind=direction,
            ),
            ScalarOutcome(
                "Total Investment Costs",
                variable_name=[var for var in cost_variables],
                function=sum_over,
                kind=direction,
            ),
            ScalarOutcome(
                "Expected Number of Deaths",
                variable_name=[var for var in casualty_variables],
                function=sum_over,
                kind=direction,
            ),
        ]

    # 5-objectives PF:
    elif problem_formulation_id == 2:
        damage_variables = []
        dike_cost_variables = []
        rfr_costs_variables = []
        evac_cost_variables = []
        casualty_variables = []

        for n in function.planning_steps:
            damage_variables.extend(
                [f"{dike}_Expected Annual Damage_{n}" for dike in function.dikelist]
            )
            dike_cost_variables.extend(
                [f"{dike}_Dike Investment Costs_{n}" for dike in function.dikelist]
            )
            rfr_costs_variables.extend([f"RfR Total Costs_{n}"])
            evac_cost_variables.extend([f"Expected Evacuation Costs_{n}"])
            casualty_variables.extend(
                [f"{dike}_Expected Number of Deaths_{n}" for dike in function.dikelist]
            )

        dike_model.outcomes = [
            ScalarOutcome(
                "Expected Annual Damage",
                variable_name=[var for var in damage_variables],
                function=sum_over,
                kind=direction,
            ),
            ScalarOutcome(
                "Dike Investment Costs",
                variable_name=[var for var in dike_cost_variables],
                function=sum_over,
                kind=direction,
            ),
            ScalarOutcome(
                "RfR Investment Costs",
                variable_name=[var for var in rfr_costs_variables],
                function=sum_over,
                kind=direction,
            ),
            ScalarOutcome(
                "Evacuation Costs",
                variable_name=[var for var in evac_cost_variables],
                function=sum_over,
                kind=direction,
            ),
            ScalarOutcome(
                "Expected Number of Deaths",
                variable_name=[var for var in casualty_variables],
                function=sum_over,
                kind=direction,
            ),
        ]

    # Disaggregate over locations:
    elif problem_formulation_id == 3:
        outcomes = []

        for dike in function.dikelist:
            for n in function.planning_steps:
                cost_variables = []
                for e in ["Expected Annual Damage", "Dike Investment Costs"]:
                    cost_variables.append(f"{dike}_{e} {n}")

                outcomes.append(
                    ScalarOutcome(
                        f"{dike} Total Costs {n}",
                        variable_name=[var for var in cost_variables],
                        function=sum_over,
                        kind=direction,
                    )
                )

                outcomes.append(
                    ScalarOutcome(
                        f"{dike}_Expected Number of Deaths {n}",
                        variable_name=f"{dike}_Expected Number of Deaths {n}",
                        function=sum_over,
                        kind=direction,
                    )
                )

        for n in function.planning_steps:
            outcomes.append(
                ScalarOutcome(
                    f"RfR Total Costs {n}",
                    variable_name=f"RfR Total Costs {n}",
                    function=sum_over,
                    kind=direction,
                )
            )
            outcomes.append(
                ScalarOutcome(
                    f"Expected Evacuation Costs {n}",
                    variable_name=f"Expected Evacuation Costs {n}",
                    function=sum_over,
                    kind=direction,
                )
            )

        dike_model.outcomes = outcomes

    # Disaggregate over time:
    elif problem_formulation_id == 4:
        outcomes = []

        for n in function.planning_steps:
            outcomes.append(
                ArrayOutcome(
                    f"Expected Annual Damage {n}",
                    variable_name=[
                        f"{dike}_Expected Annual Damage {n}" for dike in function.dikelist
                    ],
                    function=sum_over_time,
                )
            )

            outcomes.append(
                ArrayOutcome(
                    f"Dike Investment Costs {n}",
                    variable_name=[
                        f"{dike}_Dike Investment Costs {n}" for dike in function.dikelist
                    ],
                    function=sum_over_time,
                )
            )

            outcomes.append(
                ArrayOutcome(
                    f"Expected Number of Deaths {n}",
                    variable_name=[
                        f"{dike}_Expected Number of Deaths {n}" for dike in function.dikelist
                    ],
                    function=sum_over_time,
                )
            )

            outcomes.append(ArrayOutcome(f"RfR Total Costs {n}"))
            outcomes.append(ArrayOutcome(f"Expected Evacuation Costs {n}"))

        dike_model.outcomes = outcomes

    # Fully disaggregated:
    elif problem_formulation_id == 5:
        outcomes = []

        for dike in function.dikelist:
            for n in function.planning_steps:
                for entry in [
                    "Expected Annual Damage",
                    "Dike Investment Costs",
                    "Expected Number of Deaths",
                ]:

                    o = ArrayOutcome(f"{dike}_{entry} {n}")
                    outcomes.append(o)

        for n in function.planning_steps:
            outcomes.append(ArrayOutcome(f"RfR Total Costs {n}"))
            outcomes.append(ArrayOutcome(f"Expected Evacuation Costs {n}"))

        dike_model.outcomes = outcomes

    # Specific to Dike ring three
    elif problem_formulation_id == 6:
            
        damage_a3 = []
        casualties_a3 = []
        dike_costs = []
        rfr_costs = []
        evacuation_costs = []
    
        outcomes = []
        for n in function.planning_steps:

            #Damage  
            damage_a3.extend(['A.3_Expected Annual Damage {}'.format(n)])
            #Casualties
            casualties_a3.extend(['A.3_Expected Number of Deaths {}'.format(n)])
            #Costs
            for dike in function.dikelist:
                dike_costs.extend(['{}_Dike Investment Costs {}'.format(dike, n)])

            rfr_costs.extend(['RfR Total Costs {}'.format(n)])
            evacuation_costs.extend(['Expected Evacuation Costs {}'.format(n)])

        dike_model.outcomes = [
                ScalarOutcome('A3 Expected Annual Damage',
                          variable_name=[var for var in damage_a3],
                          function=sum_over, kind=ScalarOutcome.MINIMIZE),

                ScalarOutcome('A3 Aggr Expected Number of Deaths',
                          variable_name=[var for var in casualties_a3],
                          function=sum_over, kind=ScalarOutcome.MINIMIZE),

                ScalarOutcome('A3 Dike Investment Costs',
                          variable_name=[var for var in dike_costs],
                          function=sum_over, kind=ScalarOutcome.MINIMIZE),

                ScalarOutcome('Room for River Investment Costs',
                          variable_name=[var for var in rfr_costs],
                          function=sum_over, kind=ScalarOutcome.MINIMIZE),

                ScalarOutcome('Evacuation Costs',
                          variable_name=[var for var in evacuation_costs],
                          function=sum_over, kind=ScalarOutcome.MINIMIZE)]

    # Specific to Gelderland Province
    elif problem_formulation_id == 7:
        
        damage_a1_a2 = []
        damage_a3 = []
        casualties_a1_a2 = []        
        casualties_a3 = []
        dike_costs = []
        rfr_costs = []
        evacuation_costs = []
        
        outcomes = []
        for n in function.planning_steps:
            
            #Damage  
            damage_a1_a2.extend(['A.1_Expected Annual Damage {}'.format(n), 'A.2_Expected Annual Damage {}'.format(n)])
            damage_a3.extend(['A.3_Expected Annual Damage {}'.format(n)])
            #Casualties
            casualties_a1_a2.extend(['A.1_Expected Number of Deaths {}'.format(n), 'A.2_Expected Number of Deaths {}'.format(n)])
            casualties_a3.extend(['A.3_Expected Number of Deaths {}'.format(n)])
            #Costs
            for dike in function.dikelist:
                dike_costs.extend(['{}_Dike Investment Costs {}'.format(dike,n)
                                                for dike in function.dikelist])
            rfr_costs.extend(['RfR Total Costs {}'.format(n)])
            evacuation_costs.extend(['Expected Evacuation Costs {}'.format(n)])
        dike_model.outcomes = [
                    ScalarOutcome('A1_2 Aggr Expected Annual Damage',
                          variable_name=[var for var in damage_a1_a2],
                          function=sum_over, kind = ScalarOutcome.MINIMIZE),
                ScalarOutcome('A3 Expected Annual Damage',
                          variable_name=[var for var in damage_a3],
                          function=sum_over, kind=ScalarOutcome.MINIMIZE),
                ScalarOutcome('A1_2 Aggr Expected Number of Deaths',
                          variable_name=[var for var in casualties_a1_a2],
                          function=sum_over, kind=ScalarOutcome.MINIMIZE),
                          
                ScalarOutcome('A3 Aggr Expected Number of Deaths',
                          variable_name=[var for var in casualties_a3],
                          function=sum_over, kind=ScalarOutcome.MINIMIZE),
                          
                ScalarOutcome('A1_5 Dike Investment Costs',
                          variable_name=[var for var in dike_costs],
                          function=sum_over, kind=ScalarOutcome.MINIMIZE),
                          
                ScalarOutcome('Room for River Investment Costs',
                          variable_name=[var for var in rfr_costs],
                          function=sum_over, kind=ScalarOutcome.MINIMIZE),
                          
                ScalarOutcome('Evacuation Costs',
                          variable_name=[var for var in evacuation_costs],
                          function=sum_over, kind=ScalarOutcome.MINIMIZE)]
    #Specified for Overijssel
    elif problem_formulation_id == 8:
        
        damage_a4 = []
        damage_a5 = []
        casualties_a4 = []        
        casualties_a5 = []
        dike_costs = []
        rfr_costs = []
        evacuation_costs = []
        
        outcomes = []
        for n in function.planning_steps:
            
            #Damage  
            damage_a4.extend(['A.4_Expected Annual Damage {}'.format(n)])
            damage_a5.extend(['A.5_Expected Annual Damage {}'.format(n)])
            #Casualties
            casualties_a4.extend(['A.4_Expected Number of Deaths {}'.format(n)])
            casualties_a5.extend(['A.5_Expected Number of Deaths {}'.format(n)])
            #Costs
            for dike in function.dikelist:
                dike_costs.extend(['{}_Dike Investment Costs {}'.format(dike,n)
                                                for dike in function.dikelist])
            rfr_costs.extend(['RfR Total Costs {}'.format(n)])
            evacuation_costs.extend(['Expected Evacuation Costs {}'.format(n)])
        dike_model.outcomes = [
                ScalarOutcome('A4 Expected Annual Damage',
                          variable_name=[var for var in damage_a4],
                          function=sum_over, kind = ScalarOutcome.MINIMIZE),
                ScalarOutcome('A5 Expected Annual Damage',
                          variable_name=[var for var in damage_a5],
                          function=sum_over, kind=ScalarOutcome.MINIMIZE),
                ScalarOutcome('A4 Expected Number of Deaths',
                          variable_name=[var for var in casualties_a4],
                          function=sum_over, kind=ScalarOutcome.MINIMIZE),
                          
                ScalarOutcome('A5 Expected Number of Deaths',
                          variable_name=[var for var in casualties_a5],
                          function=sum_over, kind=ScalarOutcome.MINIMIZE),
                          
                ScalarOutcome('A1_5 Dike Investment Costs',
                          variable_name=[var for var in dike_costs],
                          function=sum_over, kind=ScalarOutcome.MINIMIZE),
                          
                ScalarOutcome('Room for River Investment Costs',
                          variable_name=[var for var in rfr_costs],
                          function=sum_over, kind=ScalarOutcome.MINIMIZE),
                          
                ScalarOutcome('Evacuation Costs',
                          variable_name=[var for var in evacuation_costs],
                          function=sum_over, kind=ScalarOutcome.MINIMIZE)]
        
        # 7-Objectives PF Holistic View:
    elif problem_formulation_id == 9:
  
        casualties = []
        rfr_costs = []
        evacuation_costs = []        
        gelderland_dike_cost = []
        overijssel_dike_cost = []
        gelderland_expected_damage = []
        overijssel_expected_damage = []
        
        outcomes = []
        for n in function.planning_steps:
            
            #Damage
            gelderland_expected_damage.extend(['A.1_Expected Annual Damage {}'.format(n), 'A.2_Expected Annual Damage {}'.format(n), 'A.3_Expected Annual Damage {}'.format(n)])
            overijssel_expected_damage.extend(['A.4_Expected Annual Damage {}'.format(n), 'A.5_Expected Annual Damage {}'.format(n)])
            
            #Casualties
            casualties.extend(['A.1_Expected Number of Deaths {}'.format(n), 'A.2_Expected Number of Deaths {}'.format(n), 'A.3_Expected Number of Deaths {}'.format(n),
                              'A.4_Expected Number of Deaths {}'.format(n), 'A.5_Expected Number of Deaths {}'.format(n)])
            #Costs
            for dike in function.dikelist:
                gelderland_dike_cost.extend(['{}_Dike Investment Costs {}'.format(dike,n)
                                                for dike in function.dikelist[0:len(function.dikelist)-1]])
                overijssel_dike_cost.extend(['{}_Dike Investment Costs {}'.format(dike,n)
                                                for dike in function.dikelist[4:5]])
            rfr_costs.extend(['RfR Total Costs {}'.format(n)])
            evacuation_costs.extend(['Expected Evacuation Costs {}'.format(n)])
        dike_model.outcomes = [
                    ScalarOutcome('Gelderland Expected Annual Damage',
                          variable_name=[var for var in gelderland_expected_damage],
                          function=sum_over, kind = ScalarOutcome.MINIMIZE),
                ScalarOutcome('Overijssel Expected Annual Damage',
                          variable_name=[var for var in overijssel_expected_damage],
                          function=sum_over, kind=ScalarOutcome.MINIMIZE),
                ScalarOutcome('Expected Number of Deaths',
                          variable_name=[var for var in casualties],
                          function=sum_over, kind=ScalarOutcome.MINIMIZE),
                          
                ScalarOutcome('Gelderland Dike Cost',
                          variable_name=[var for var in gelderland_dike_cost],
                          function=sum_over, kind=ScalarOutcome.MINIMIZE),
                          
                ScalarOutcome('Overijssel Dike Cost',
                          variable_name=[var for var in overijssel_dike_cost],
                          function=sum_over, kind=ScalarOutcome.MINIMIZE),
                          
                ScalarOutcome('Room for River Investment Costs',
                          variable_name=[var for var in rfr_costs],
                          function=sum_over, kind=ScalarOutcome.MINIMIZE),
                          
                ScalarOutcome('Evacuation Costs',
                          variable_name=[var for var in evacuation_costs],
                          function=sum_over, kind=ScalarOutcome.MINIMIZE)]
    else:
        raise TypeError("unknown identifier")

    return dike_model, function.planning_steps

if __name__ == "__main__":
    get_model_for_problem_formulation(3)
