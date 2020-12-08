#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

from typing import Tuple

import miplearn.log
import numpy as np
import pandas as pd
from egret.models.unit_commitment import (
    create_tight_unit_commitment_model,
    _save_uc_results,
)
from miplearn import (
    LearningSolver,
    Instance,
    GurobiPyomoSolver,
    PrimalSolutionComponent,
    AdaptiveClassifier,
)
from miplearn.classifiers.counting import CountingClassifier
from prescient.data.simulation_state import SimulationState
from prescient.engine.abstract_types import OperationsModel, RucModel
from prescient.engine.data_extractors import ScedDataExtractor, RucDataExtractor
from prescient.engine.egret import EgretEngine
from prescient.engine.modeling_engine import ModelingEngine, ForecastErrorMethod
from prescient.simulator import Options
from sklearn.neighbors import KNeighborsClassifier

miplearn.log.setup_logger()


class RucLearningInstance(Instance):
    def __init__(self, ruc_model: RucModel):
        self.ruc_model = ruc_model
        self.pyomo_model = create_tight_unit_commitment_model(self.ruc_model)

    def to_model(self):
        return self.pyomo_model

    def get_instance_features(self):
        return np.array(list(self.pyomo_model.TotalDemand.values()))

    def get_variable_category(self, var_name, idx):
        if var_name in ["UnitStart", "UnitStop"]:
            return (var_name, idx)


class MiplearnEngine(ModelingEngine):
    def __init__(self):
        self.egret = EgretEngine()
        self.solver = LearningSolver(
            solver=GurobiPyomoSolver(),
            solve_lp_first=True,
            components=[PrimalSolutionComponent(
                classifier=AdaptiveClassifier(
                    candidates={
                        "knn": {
                            "classifier": KNeighborsClassifier(n_neighbors=5),
                            "min samples": 5,
                        },
                        "counting": {
                            "classifier": CountingClassifier(),
                            "min samples": 0,
                        },
                    },
                ),
            )],
        )
        self.solved_instances = []

    def initialize(self, options: Options) -> None:
        self.egret.initialize(options)

    def solve_deterministic_ruc(
            self,
            options: Options,
            ruc_instance: RucModel,
            uc_date: str,
            uc_hour: int,
    ) -> RucModel:
        # Convert raw RUC instance into MIPLearn instance
        instance = RucLearningInstance(ruc_instance)

        # Solve instance using ML
        self.solver.solve(instance, tee=True)

        # Evaluate ML accuracy on the instance we just solved
        comp = self.solver.components["PrimalSolutionComponent"]
        ev = comp.evaluate([instance])
        print("\nFix one:")
        print(pd.DataFrame(ev["Fix one"]).mean(axis=1))
        print("\nFix zero:")
        print(pd.DataFrame(ev["Fix zero"]).mean(axis=1))

        # Add solved instance to training dataset and retrain ML components
        self.solved_instances += [instance]
        self.solver.fit(self.solved_instances)

        return _save_uc_results(instance.pyomo_model, relaxed=False)

    def create_deterministic_ruc(
            self,
            options: Options,
            uc_date: str,
            uc_hour: int,
            current_state: SimulationState,
            output_ruc_initial_conditions: bool,
            ruc_horizon: int,
            use_next_day_data: bool,
    ) -> RucModel:
        return self.egret.create_deterministic_ruc(
            options,
            uc_date,
            uc_hour,
            current_state,
            output_ruc_initial_conditions,
            ruc_horizon,
            use_next_day_data,
        )

    def create_simulation_actuals(
            self,
            options: Options,
            uc_date: str,
            uc_hour: int,
    ) -> RucModel:
        return self.egret.create_simulation_actuals(options, uc_date, uc_hour)

    def create_sced_instance(
            self,
            options: Options,
            current_state: SimulationState,
            hours_in_objective: int = 1,
            sced_horizon: int = 24,
            forecast_error_method: ForecastErrorMethod = ForecastErrorMethod.PRESCIENT,
            write_sced_instance: bool = False,
            lp_filename: str = None,
    ) -> OperationsModel:
        return self.egret.create_sced_instance(
            options,
            current_state,
            hours_in_objective,
            sced_horizon,
            forecast_error_method,
            write_sced_instance,
            lp_filename,
        )

    def solve_sced_instance(
            self,
            options: Options,
            sced_instance: OperationsModel,
            output_initial_conditions: bool = False,
            output_demands: bool = False,
            lp_filename: str = None,
    ) -> Tuple[OperationsModel, float]:
        return self.egret.solve_sced_instance(
            options,
            sced_instance,
            output_initial_conditions,
            output_demands,
            lp_filename,
        )

    def create_and_solve_lmp(
            self,
            options: Options,
            sced_instance: OperationsModel,
    ) -> OperationsModel:
        return self.egret.create_and_solve_lmp(options, sced_instance)

    @property
    def ruc_data_extractor(self) -> RucDataExtractor:
        return self.egret.ruc_data_extractor

    @property
    def operations_data_extractor(self) -> ScedDataExtractor:
        return self.egret.operations_data_extractor
