import pickle
import copy

import simbench as sb
import pandapower as pp
import numpy as np
import pandas as pd


from electric_gym.grid_model.controllers.storage_controller import StorageController
from electric_gym.grid_model.controllers.ev_charger_controller import EVChargerController
from electric_gym.grid_model.controllers.heat_pump_controller import HeatPumpController
from electric_gym.grid_model.controllers.pv_controller import PVController

from electric_gym.grid_model.grid_utils import (
    get_simbench_grid,
    min_max_normalize,
    extract_feeder,
    fix_bus_names,
    create_ts_dataframe,
)


class SingleFeederGridManager:

    def __init__(self, power_scaling, control_p, control_q, control_q_shielding):

        self.power_scaling = power_scaling
        self.control_p = control_p
        self.control_q = control_q
        self.control_q_shielding = control_q_shielding

        self.reference_grid = get_simbench_grid()
        sb_grid = copy.deepcopy(self.reference_grid)

        # Fix and print bus names
        sb_grid.bus["name"] = fix_bus_names(sb_grid.bus["name"])

        # List of bus IDs to be removed to obtain the single feeder zone
        remove_bus_list = [
            8,
            11,
            10,
            3,  # First feeder
            2,
            9,
            13,  # Second feeder
            1,  # Fourth feeder
        ]
        # List of keys corresponding to the tables that need to be modified.
        ppnet_keys_list = [
            "bus",
            "load",
            "storage",
            "sgen",
            "ext_grid",
            "line",
            "trafo",
            "bus_geodata",
        ]
        # Extract the feeder from the grid by removing the specified buses
        self.grid = extract_feeder(sb_grid, remove_bus_list, ppnet_keys_list)

        # Add additional PV units to the grid
        self.add_pv()

        # Add additional HP units to the grid
        self.add_hp()

        # Create the load profiles from the reference grid
        self.create_load_profiles()

        # Create the additional storages
        self.create_storages()

        self.storage_controllers = []
        self.pv_controllers = []
        self.heat_pump_controllers = []
        self.ev_controllers = []

        for storage_id in self.grid.storage.index:
            self.storage_controllers.append(
                StorageController(
                    id=storage_id,
                    storage=self.grid.storage.loc[storage_id],
                    power_scaling=power_scaling,
                    control_p=control_p,
                    control_q=control_q,
                    control_q_shielding=control_q_shielding,
                )
            )

        for sgen_id in self.grid.sgen.index:
            self.pv_controllers.append(
                PVController(
                    id=sgen_id,
                    pv=self.grid.sgen.loc[sgen_id],
                    power_scaling=power_scaling,
                    control_q=control_q,
                    control_q_shielding=control_q_shielding,
                )
            )

        for load_id in self.grid.load.index:
            load = self.grid.load.loc[load_id]
            if load.profile.startswith("Soil") or load.profile.startswith("Air"):
                self.heat_pump_controllers.append(
                    HeatPumpController(
                        id=load_id,
                        load=load,
                        power_scaling=power_scaling,
                        control_p=control_p,
                    )
                )

        self.load_car_distributions()
        for i in range(2, 7):
            ev_id = pp.create_storage(
                self.grid,
                bus=i,
                p_mw=0,
                q_mvar=0,
                sn_mva=0.008 * self.power_scaling,
                soc_percent=0.5,
                min_e_mwh=0,
                max_e_mwh=0.075,
                max_p_mw=0.008,
                min_p_mw=0,
                scaling=self.power_scaling,
                in_service=True,
                efficiency_percent=0.95,
                name="EV Charger Bus 3",
            )
            self.ev_controllers.append(
                EVChargerController(
                    id=ev_id,
                    storage=self.grid.storage.loc[ev_id],
                    power_scaling=power_scaling,
                    control_p=control_p,
                    arrival_probabilities=self.ev_arrival_probabilities,
                    departure_probabilities=self.ev_departure_probabilities,
                )
            )

        # The indices of the loads that are not controllable (Household and uncontrolled EV chargers) and the heat pumps
        self.hp_indices = []
        self.pure_loads = []
        for index in self.grid.load.index:
            load = self.grid.load.loc[index]
            if load.profile.startswith("Soil") or load.profile.startswith("Air"):
                self.hp_indices.append(index)
            else:
                self.pure_loads.append(index)

        self.storages_count = len(self.storage_controllers)
        self.pv_count = len(self.pv_controllers)
        self.ev_count = len(self.ev_controllers)
        self.hp_count = len(self.heat_pump_controllers)

        if self.control_p:
            self.p_controllers_count = self.ev_count + self.storages_count + self.hp_count
        else:
            self.p_controllers_count = 0
        if self.control_q:
            self.q_controllers_count = self.pv_count + self.storages_count
        else:
            self.q_controllers_count = 0

        self.calculate_boundary_values()

        self.grid.line.at[2, "length_km"] = self.grid.line.at[2, "length_km"] * 10
        self.grid.line.at[4, "length_km"] = self.grid.line.at[4, "length_km"] * 10

    def step(self, actions=None, timestep=0, random_numbers=None):
        """
        Run a simulation step for the grid, applying the given actions and updating the grid state.
        """
        penalties = self.run_controllers(actions=actions, timestep=timestep, random_numbers=random_numbers)
        aborted = self.run_powerflow(timestep)
        if self.control_q_shielding:
            self.run_shielding_controllers(self.grid)
            aborted = self.run_powerflow(timestep)
        return aborted, penalties

    def reset(self, timestep, np_random):
        """
        Reset the grid state to the initial state at the given timestep.
        """
        self.assign_profiles(timestep)
        self.np_random = np_random

        for storage_controller in self.storage_controllers:
            storage_controller.reset(grid_state=self.grid)
        for ev_controller in self.ev_controllers:
            ev_controller.reset(grid_state=self.grid)
        for hp_controller in self.heat_pump_controllers:
            hp_controller.reset(grid_state=self.grid)
        for pv_controller in self.pv_controllers:
            pv_controller.reset(grid_state=self.grid)

        self.run_powerflow(timestep)

    def run_shielding_controllers(self, grid_state):
        """
        Runs only the controllers supporting the VDE piecewise linear control scheme.
        """
        for bes in self.storage_controllers:
            bes.vde_adjust(self.grid, "voltage_reactive")
        for pv in self.pv_controllers:
            pv.voltage_reactive_q_control(grid_state)

    def run_controllers(self, actions, timestep, random_numbers):
        """
        Run the controllers for each component in the grid.
        Returns the penalties incurred by each controller type (sum of all controllers of that type).
        """

        storage_offset = 0
        ev_offset = self.storages_count
        hp_offset = self.storages_count + self.ev_count
        pv_offset = self.storages_count

        if self.control_p:
            p_actions = actions[: self.p_controllers_count]
        else:
            p_actions = [0] * (self.storages_count + self.ev_count + self.hp_count)
        if self.control_q:
            q_actions = actions[self.p_controllers_count :]
        else:
            q_actions = [0] * (self.storages_count + self.pv_count)

        bes_penalties = 0
        ev_penalties = 0
        pv_penalties = 0
        hp_penalties = 0

        for index, storage_controller in enumerate(self.storage_controllers):
            bes_penalty = storage_controller.run(
                grid_state=self.grid,
                p_action=p_actions[index + storage_offset],
                q_action=q_actions[index + storage_offset],
                timestep=timestep,
            )
            bes_penalties += bes_penalty

        for index, ev_controller in enumerate(self.ev_controllers):
            ev_penalty = ev_controller.run(
                grid_state=self.grid,
                p_action=p_actions[index + ev_offset],
                timestep=timestep,
                random_numbers=random_numbers[index],
            )
            ev_penalties += ev_penalty

        for index, hp_controller in enumerate(self.heat_pump_controllers):
            hp_penalty = hp_controller.run(
                grid_state=self.grid,
                p_action=p_actions[index + hp_offset],
                timestep=timestep,
            )
            hp_penalties += hp_penalty

        for index, pv_controller in enumerate(self.pv_controllers):
            pv_penalty = pv_controller.run(
                grid_state=self.grid,
                q_action=q_actions[index + pv_offset],
                timestep=timestep,
            )
            pv_penalties += pv_penalty

        return {
            "battery_penalty": bes_penalties,
            "charger_penalty": ev_penalties,
            "pv_penalty": pv_penalties,
            "hp_penalty": hp_penalties,
        }

    def run_powerflow(self, t):
        """
        Run a power flow calculation for the grid.
        Returns True if the power flow calculation failed to converge.
        """
        try:
            pp.runpp(
                self.grid,
                init="results",
                calculate_voltage_angles=False,
                tolerance_mva=1e-6,
            )
        except:
            print("Failed to converge at time step: ", t)
            print(pp.diagnostic(self.grid))
            return True
        return False

    def get_controllers_count(self):
        """
        Returns a tuple with the number of controllers for each component type.
        """
        return self.storages_count, self.pv_count, self.ev_count, self.hp_count

    def get_control_space(self):
        """
        Returns the dimensions of the action space for the agent.
        """
        return self.p_controllers_count, self.q_controllers_count

    def get_current_power(self):
        """
        Returns the current power consumption and generation for each component type.
        """
        loads = self.grid.load.loc[self.pure_loads, "p_mw"].values
        sgen = self.grid.sgen.loc[:, "p_mw"].values
        storage = np.array([sc.p_mw for sc in self.storage_controllers], dtype=np.float64)
        ev_p = np.array([evc.p_mw for evc in self.ev_controllers], dtype=np.float64)
        hp_p = np.array([hpc.p_mw for hpc in self.heat_pump_controllers], dtype=np.float64)

        return loads, sgen, storage, ev_p, hp_p

    def get_storage_soc(self):
        return np.array([sc.soc_percent for sc in self.storage_controllers])

    def get_ev_soc(self):
        return np.array([ev.soc_percent for ev in self.ev_controllers])

    def get_ev_present(self):
        return np.array([ev.at_home for ev in self.ev_controllers])

    def get_heat_storage(self):
        return np.array([hpc.storage() for hpc in self.heat_pump_controllers])

    def get_prev_heat_pump_power(self):
        return np.array(
            [hpc.get_prev_load() for hpc in self.heat_pump_controllers],
            dtype=np.float64,
        )

    def get_line_loading(self):
        return self.grid.res_line.loading_percent.values

    def get_line_losses(self):
        return self.grid.res_line.pl_mw.values

    def get_transformer_losses(self):
        return self.grid.res_trafo.pl_mw.values

    def get_transformer_loading(self):
        return self.grid.res_trafo.loading_percent.values

    def get_ext_power(self):
        return self.grid.res_ext_grid.p_mw.values

    def get_voltage_pu(self, normalize=False):
        voltages = self.grid.res_bus.vm_pu.values
        # Remove the bus connected to the external grid
        voltages = voltages[1:]
        if normalize:
            min_max_normalizer = np.vectorize(min_max_normalize)
            voltages = min_max_normalizer(voltages, 0.75, 1.25)
        return voltages

    def get_pv_reactive_power(self):
        return self.grid.sgen.loc[:, "q_mvar"].values

    def get_pv_curtailment(self):
        return np.array([pv.get_curtailment() for pv in self.pv_controllers])

    def get_max_net_power(self):
        """
        The theoretical maximum power that can be generated by the grid

        """
        return self.max_sgen + self.max_storage_p + self.max_ev_p - self.min_load_p

    def get_min_net_power(self):
        """
        The theoretical maximum power that can be consumed by the grid
        """
        return self.min_sgen - self.max_storage_p - self.max_load_p - self.max_ev_p

    def get_controllable_storage_power(self):
        """
        Returns the maximum power that can be controlled by the storage units.
        """
        max_p = [storage.max_p_mw for storage in self.storage_controllers]
        max_e = [storage.max_e_mwh for storage in self.storage_controllers]
        return (max_p, max_e)

    def get_controllable_ev_power(self):
        """
        Returns the maximum power that can be controlled by the EV chargers.
        """
        max_p = [ev.max_p_mw for ev in self.ev_controllers]
        max_e = [ev.max_e_mwh for ev in self.ev_controllers]
        return max_p, max_e

    def get_controllable_hp_power(self):
        """
        Returns the maximum power that can be controlled by the heat pumps.
        """
        return [hp.max_p_mw for hp in self.heat_pump_controllers]

    def load_car_distributions(self):
        with open("./data/car_distributions.csv", "r") as f:
            data = pd.read_csv(f)
        self.ev_arrival_probabilities = data["arrival"]
        self.ev_departure_probabilities = data["leave"]

    def calculate_boundary_values(self):
        self.min_load_p = max(self.active_profiles.min())
        self.max_load_p = max(self.active_profiles.max())

        self.max_hp_p = max(hp.max_p_mw for hp in self.heat_pump_controllers)

        self.min_load_q = min(self.reactive_profiles.min())
        self.max_load_q = max(self.reactive_profiles.max())

        self.min_sgen = min(self.sgen_profiles.min())
        self.max_sgen = max(self.sgen_profiles.max())

        self.max_storage_p = max(storage.max_p_mw for storage in self.storage_controllers)

        self.max_ev_p = max(ev.max_p_mw for ev in self.ev_controllers)
        self.max_e_mwh = max(storage.max_e_mwh for storage in self.storage_controllers)

    def assign_profiles(self, timestep):
        for i in range(self.grid.load.shape[0]):
            self.grid.load.at[i, "p_mw"] = self.active_profiles[i][timestep] * self.power_scaling
            self.grid.load.at[i, "q_mvar"] = self.reactive_profiles[i][timestep] * self.power_scaling
        for i in range(self.grid.sgen.shape[0]):
            self.grid.sgen.at[i, "p_mw"] = self.sgen_profiles[i][timestep] * self.power_scaling

    def get_uncontrolled_power(self, horizon, timestep, normalize=False):
        load = self.active_profiles[self.pure_loads][timestep : timestep + horizon]
        sgen = self.sgen_profiles[:][timestep : timestep + horizon]
        hp_p = self.active_profiles[self.hp_indices][timestep : timestep + horizon]

        min_max_normalizer = np.vectorize(min_max_normalize)
        if normalize:
            load = min_max_normalizer(load, 0, self.max_load_p)
            sgen = min_max_normalizer(sgen, 0, self.max_sgen)
            hp_p = min_max_normalizer(hp_p, 0, self.max_hp_p)

        return load, sgen, hp_p

    def get_controlled_power(
        self,
    ):
        storage = [controller.p_mw for controller in self.storage_controllers]
        ev_p = [controller.p_mw for controller in self.ev_controllers]
        hp_p = [controller.p_mw for controller in self.heat_pump_controllers]

        return storage, ev_p, hp_p

    def get_observation_space(self):
        """
        Returns the dimensions of the observation space for the agent.
        """
        pv_obs_dim = self.pv_count
        load_obs_dim = 7
        ev_obs_dim = self.ev_count
        hp_obs_dim = self.hp_count
        storage_obs_dim = self.storages_count

        return pv_obs_dim, load_obs_dim, storage_obs_dim, ev_obs_dim, hp_obs_dim

    def create_load_profiles(self):
        time_series_profiles = sb.get_absolute_values(self.reference_grid, profiles_instead_of_study_cases=True)

        ts_load_q = time_series_profiles[("load", "q_mvar")]
        ts_load_p = time_series_profiles[("load", "p_mw")]
        ts_sgen_p = time_series_profiles[("sgen", "p_mw")]

        ts_load_q = create_ts_dataframe(self.grid, "load", ts_load_q)
        ts_load_p = create_ts_dataframe(self.grid, "load", ts_load_p)
        ts_sgen_p = create_ts_dataframe(self.grid, "sgen", ts_sgen_p)

        self.active_profiles = ts_load_p
        self.reactive_profiles = ts_load_q
        self.sgen_profiles = ts_sgen_p

    def add_pv(self):
        pv_0 = self.reference_grid.sgen.loc[4]
        new_sgen_id = pp.create_sgen(
            self.grid,
            bus=3,
            p_mw=0,
            q_mvar=0,
            sn_mva=pv_0["sn_mva"] * self.power_scaling,
            name=pv_0["name"],
            min_p_mw=pv_0["min_p_mw"] * self.power_scaling,
            max_p_mw=pv_0["max_p_mw"] * self.power_scaling,
        )

        pv_1 = self.reference_grid.sgen.loc[1]

        new_sgen_id = pp.create_sgen(
            self.grid,
            bus=5,
            p_mw=0,
            q_mvar=0,
            sn_mva=pv_1["sn_mva"] * self.power_scaling,
            name=pv_1["name"],
            min_p_mw=pv_1["min_p_mw"] * self.power_scaling,
            max_p_mw=pv_1["max_p_mw"] * self.power_scaling,
        )

        pv_2 = self.reference_grid.sgen.loc[5]

        new_sgen_id = pp.create_sgen(
            self.grid,
            bus=6,
            p_mw=0,
            q_mvar=0,
            sn_mva=pv_2["sn_mva"] * self.power_scaling,
            name=pv_2["name"],
            min_p_mw=pv_2["min_p_mw"] * self.power_scaling,
            max_p_mw=pv_2["max_p_mw"] * self.power_scaling,
        )

    def add_hp(self):
        hp_load = self.reference_grid.load.loc[25]
        pp.create_load(
            net=self.grid,
            bus=4,
            p_mw=0,
            q_mvar=0,
            sn_mva=hp_load["sn_mva"],
            name=hp_load["name"],
            min_p_mw=hp_load["min_p_mw"],
            max_p_mw=hp_load["max_p_mw"],
            profile=hp_load["profile"],
        )

    def create_storages(self):

        storage = self.reference_grid.storage.loc[1]
        pp.create_storage(
            self.grid,
            bus=2,
            p_mw=0,
            q_mvar=0,
            sn_mva=storage["sn_mva"] * self.power_scaling,
            soc_percent=0.5,
            min_e_mwh=storage["min_e_mwh"],
            max_e_mwh=storage["max_e_mwh"],
            max_p_mw=storage["max_p_mw"] * self.power_scaling,
            min_p_mw=storage["min_p_mw"] * self.power_scaling,
            scaling=self.power_scaling,
            in_service=True,
            efficiency_percent=storage["efficiency_percent"],
            name=storage["name"],
        )

        storage = self.reference_grid.storage.loc[2]
        pp.create_storage(
            self.grid,
            bus=4,
            p_mw=0,
            q_mvar=0,
            sn_mva=storage["sn_mva"] * self.power_scaling,
            soc_percent=0.5,
            min_e_mwh=storage["min_e_mwh"],
            max_e_mwh=storage["max_e_mwh"],
            max_p_mw=storage["max_p_mw"] * self.power_scaling,
            min_p_mw=storage["min_p_mw"] * self.power_scaling,
            scaling=self.power_scaling,
            in_service=True,
            efficiency_percent=storage["efficiency_percent"],
            name=storage["name"],
        )
