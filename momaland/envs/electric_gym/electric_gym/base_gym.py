import functools
import random
import numpy as np
import pandas as pd

from typing_extensions import override

from gymnasium import spaces
from gymnasium.spaces import Box, Dict
from gymnasium.utils import EzPickle
from momaland.utils.env import MOParallelEnv

from electric_gym.grid_model.single_feeder_grid_manager import SingleFeederGridManager


def env(**kwargs):
    return BaseElectricGym(**kwargs)


def raw_env(**kwargs):
    return BaseElectricGym(**kwargs)


class BaseElectricGym(MOParallelEnv, EzPickle):

    def __init__(
        self,
        render_mode=None,
        num_days=7,
        normalize=True,
        forecast_horizon=8,
        power_scaling=1,
        observation_space="box",
        test_split=False,
        apply_noise=True,
    ):
        """
        Base class for the electric grid environment.
        args:

        render_mode: str, optional
            The mode for rendering the environment. Currently no rendering is supported.
        num_days: int, optional
            The number of days to simulate in the environment.
        normalize: bool, optional
            f True, the observations are normalized.
        power_scaling: float, optional
            A scaling factor for the power values in the environment.
        economic_scaling: float, optional
            A scaling factor for the economic rewards in the environment.
        voltage_scaling: float, optional
            A scaling factor for the voltage rewards in the environment.
        load_scaling: float, optional
            A scaling factor for the load rewards in the environment.
        energy_loss_scaling: float, optional
            A scaling factor for the energy loss rewards in the environment.
        ev_flexibility_scaling: float, optional
            A scaling factor for the EV flexibility rewards in the environment.
        curtailment_cost_scaling: float, optional
            A scaling factor for the curtailment rewards in the environment.
        use_case: str, optional
            The use case for the environment. Options are "economic", "voltage", "combined"
        forecast_horizon: int, optional
            The forecast horizon (load, price, pv forecast) available in the observation space, number of timesteps.
        grid: str, optional
            The grid to simulate. Options are "single_feeder" and "household".
        observation_space: str, optional
            The observation space type. Options are "box" and "dict".
        mo: bool, optional
            If True, the environment is multi-objective, returing a vector of rewards.
        test_split: bool, optional
            If True, the environment is in test mode, using a predefined test split.
        rule_based_voltage: bool, optional
            Special case for the voltage use case, where the environment uses only the rule-based voltage controller.
        apply_noise: bool, optional
            If True, the environment applies noise to the observations.
        """
        EzPickle.__init__(
            self,
            render_mode=render_mode,
            num_days=num_days,
            normalize=normalize,
            forecast_horizon=forecast_horizon,
            power_scaling=power_scaling,
            observation_space=observation_space,
            test_split=test_split,
            apply_noise=apply_noise,
        )

        self.forecast_horizon = forecast_horizon
        self.power_scaling = power_scaling
        self.test_split = test_split
        self.test_counter = 0
        self.apply_noise = apply_noise

        self.observation_space_type = observation_space

        self.control_p = True
        self.control_q = True
        self.control_q_shielding = False

        self.reward_dim = 3

        self.gridMgr = self.create_grid()

        self.possible_agents = [i for i in range(2, 2 + self.gridMgr.get_controllers_count() + 1)]

        """
        Actions are:

        action[0]: BES P control
        action[1]: BES Q control
        action[2]: EV P control
        action[3]: HP P control
        action[4]: PV Q control
        """
        self._action_spaces = {agent: spaces.Box(low=-1, high=1, shape=(5,)) for agent in self.possible_agents}

        base_observation_space = {
            "pv_power": spaces.Box(
                low=0,
                high=1,
                shape=(forecast_horizon,),
            ),
            "load_power": spaces.Box(
                low=0,
                high=1,
                shape=(forecast_horizon,),
            ),
            "heating_power": spaces.Box(
                low=0,
                high=1,
                shape=(forecast_horizon,),
            ),
            "heat_storage": spaces.Box(
                low=-1,
                high=1,
                shape=(1,),
            ),
            "soc_percent": spaces.Box(
                low=0,
                high=1,
                shape=(1,),
            ),
            "ev_soc_percent": spaces.Box(
                low=0,
                high=1,
                shape=(1,),
            ),
            "ev_present": spaces.Box(low=0, high=1, shape=(1,)),
            "time_of_day": spaces.Box(low=0, high=1, shape=(1,)),
            "date": spaces.Box(low=0, high=1, shape=(1,)),
        }

        base_observation_space["price"] = spaces.Box(low=0, high=1, shape=(forecast_horizon,))
        base_observation_space["voltages"] = spaces.Box(low=0, high=1, shape=(1,))
        base_observation_space["line_loading"] = spaces.Box(low=0, high=1, shape=(4,))
        base_observation_space["transformer_loading"] = spaces.Box(low=0, high=1, shape=(1,))

        self._observation_spaces = {agent: base_observation_space for agent in self.possible_agents}

        """
        Rewards are:
        base_reward_space[0]: Economic reward
        base_reward_space[1]: Flexibility reward
        base_reward_space[2]: Voltage reward
        base_reward_space[3]: PV curtailment cost
        base_reward_space[4]: Loading cost
        base_reward_space[5]: Soft constraint penalty
        """
        base_reward_space = Box(low=-1, high=1, shape=(6,))

        self._reward_spaces = {agent: base_reward_space for agent in self.possible_agents}

        self.num_days = num_days
        self.power_scaling = power_scaling
        self.t = 0

        days = [i for i in range(3, 356, 7)]
        eval_days = days[3::4]
        eval_days = eval_days[:-1]
        self.test_split_days = eval_days
        self.train_split = list(filter(lambda x: x not in eval_days, days))

        with open("./data/spotmarket_reduced_quarters_2023.csv", "r") as f:
            self.prices = pd.read_csv(f, index_col=0)

        self.max_net_power = self.gridMgr.get_max_net_power()
        self.min_net_power = self.gridMgr.get_min_net_power()
        self.max_price = self.prices["price_clipped"].max()

        self.max_profit = self.max_net_power * self.max_price
        self.min_profit = self.min_net_power * self.max_price

        self.net_power = True
        self.normalize = normalize

        self.ev_target = 0.5

    @functools.lru_cache(maxsize=None)
    @override
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    @override
    def action_space(self, agent):
        return self._action_spaces[agent]

    @functools.lru_cache(maxsize=None)
    @override
    def reward_space(self, agent):
        return self._reward_spaces[agent]

    @override
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        self.agents = self.possible_agents[:]
        if self.test_split:
            self.start = self.test_split_days[self.test_counter] * 96
            self.test_counter += 1
            self.test_counter = self.test_counter % len(self.test_split_Days)
        else:
            self.start = self.np_random.choice(self.train_split) * 96
        self.t = self.start

        self.gridMgr.reset(timestep=self.t, np_random=self.np_random)

        observation = {agent: self._get_observation(agent) for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: self.get_info(agent) for agent in self.agents}
        self.rewards = {agent: np.zeros(6) for agent in self.agents}

        return observation, self.infos

    @override
    def step(self, action):
        """Steps in the environment.

        Args:
            actions: a dict of actions, keyed by agent names

        Returns: a tuple containing the following items in order:
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """

        terminated, penalties = self._take_action(action)

        if terminated:
            print("PP did not converge", self.t)
        rewards = {agent: self._get_rewards(agent) for agent in self.agents}
        self.t += 1
        info = self._get_info(penalties=penalties, rewards=rewards)
        self.gridMgr.assign_profiles(timestep=self.t)
        observation = {agent: self._get_observation(agent) for agent in self.agents}
        truncated = self._check_if_done()
        if self.mo:
            penalty_sum = sum([val for key, val in penalties.items()])
            reward = np.array([val for key, val in rewards.items()])
            reward = np.append(reward, -penalty_sum * 2)
        else:
            penalty_sum = sum([val for key, val in penalties.items()]) * 2
            reward_sum = sum([val for key, val in rewards.items()])
            reward = reward_sum - penalty_sum
        if terminated:
            print("Terminated because PP did not converge")
            print("action", action)
            print("reward", reward)
            print("penalties", penalties)
            print("observation", observation)
        return observation, reward, terminated, truncated, info

    def create_grid(self):
        gridMgr = SingleFeederGridManager(
            power_scaling=self.power_scaling,
            control_q=self.control_q,
            control_p=self.control_p,
            control_q_shielding=self.control_q_shielding,
        )

        return gridMgr

    def _get_observation(self, agent):

        # account for the higher level grid connection busses (1 bus in front and 1 after the transformer)
        price = self._current_price_forecast(horizon=self.forecast_horizon, normalize=self.normalize)

        load_power, pv_power, hp_power = self.gridMgr.get_bus_uncontrolled_power(
            bus=agent,
            horizon=self.forecast_horizon,
            normalize=self.normalize,
            timestep=self.t,
        )

        soc_percent = self.gridMgr.get_bus_storage_soc(agent)
        ev_soc_percent = self.gridMgr.get_bust_ev_soc(agent)
        ev_present = self.gridMgr.get_bus_ev_present(agent)
        heat_storage = self.gridMgr.get_bus_heat_storage(agent)

        time = self.t % 96
        time_normalized = time / 96

        date = self.t // 96
        date_normalized = date / 365

        """
        If normalized to [0,1] apply 1% noise, else apply 0.001 which is equivalent to 1kW in the original scale.
        """

        if self.normalize:
            load_noise = self.np_random.normal(0, 0.01, load_power.shape)
            pv_noise = self.np_random.normal(0, 0.01, pv_power.shape)
            hp_noise = self.np_random.normal(0, 0.01, hp_power.shape)
        else:
            load_noise = self.np_random.normal(0, 0.001, load_power.shape)
            pv_noise = self.np_random.normal(0, 0.001, pv_power.shape)
            hp_noise = self.np_random.normal(0, 0.001, hp_power.shape)

        load_power += load_noise
        pv_power += pv_noise
        hp_power += hp_noise

        load_power = np.clip(load_power, 0, 1)
        pv_power = np.clip(pv_power, 0, 1)
        hp_power = np.clip(hp_power, 0, 1)

        base_observation = {
            "pv_power": np.array(pv_power, dtype=np.float32),
            "load_power": np.array(load_power, dtype=np.float32),
            "heating_power": np.array(hp_power, dtype=np.float32),
            "heat_storage": np.array(heat_storage, dtype=np.float32),
            "soc_percent": np.array(soc_percent, dtype=np.float32),
            "ev_soc_percent": np.array(ev_soc_percent, dtype=np.float32),
            "ev_present": np.array(ev_present, dtype=np.int8),
            "time_of_day": np.array([time_normalized], dtype=np.float32),
            "date": np.array([date_normalized], dtype=np.float32),
        }

        if self.use_case == "economic":

            price_noise = self.np_random.normal(0, 0.01, price.shape)
            price += price_noise

            base_observation["price"] = np.array(price, dtype=np.float32)
        if self.use_case == "voltage":
            voltages = self.gridMgr.get_bus_voltage_pu(bus=agent, normalize=self.normalize)

            voltage_noise = self.np_random.normal(0, 0.01, voltages.shape)
            voltages += voltage_noise
            voltages = np.clip(voltages, 0, 1)

            base_observation["voltages"] = np.array(voltages, dtype=np.float32)
        if self.use_case == "combined":
            price_noise = self.np_random.normal(0, 0.01, price.shape)
            price += price_noise
            base_observation["price"] = np.array(price, dtype=np.float32)

            voltages = self.gridMgr.get_bus_voltage_pu(bus=agent, normalize=self.normalize)
            line_loading = self.gridMgr.get_bus_line_loading()
            trafo_loading = self.gridMgr.get_transformer_loading()

            line_loading = self._normalize(line_loading, 0, 100)
            trafo_loading = self._normalize(trafo_loading, 0, 100)

            voltage_noise = self.np_random.normal(0, 0.01, voltages.shape)
            voltages += voltage_noise
            voltages = np.clip(voltages, 0, 1)

            base_observation["voltages"] = np.array(voltages, dtype=np.float32)
            base_observation["line_loading"] = np.array(line_loading, dtype=np.float32)
            base_observation["transformer_loading"] = np.array(trafo_loading, dtype=np.float32)

        if self.observation_space_type == "dict":
            return base_observation

        flattened_observation = np.concatenate([value.flatten() for value in base_observation.values()])
        return flattened_observation

    def _get_rewards(self, agent):
        rewards = {}
        if self.use_case == "economic":
            rewards["price"] = self._get_price_reward(agent)
            rewards["ev_flexibility"] = self._get_ev_flexibility_reward(agent)

        elif self.use_case == "voltage":
            rewards["voltage"] = self._get_voltage_reward(agent)
            rewards["curtailment"] = self._get_pv_curtailment_cost(agent)

        if self.use_case == "combined":
            rewards["price"] = self._get_price_reward(agent)
            rewards["ev_flexibility"] = self._get_ev_flexibility_reward(agent)
            rewards["loading"] = self._get_loading_reward(agent)
            rewards["voltage"] = self._get_voltage_reward(agent)
            rewards["curtailment"] = self._get_pv_curtailment_cost(agent)

        return rewards

    def _get_price_reward(self, agent):
        """
        Returns the economic reward based on the current price and power consumption.
        """
        load_power, pv_power, storage_power, charging_point_power, hp_power = self.gridMgr.get_current_power()
        net_power = pv_power.sum() - sum(storage_power) - load_power.sum() - sum(charging_point_power) - sum(hp_power)
        cost = net_power * self._current_price() * 0.25
        return cost

    def _get_loading_reward(self, agent):
        """
        Returns a cost based on the transformer and line component loading.
        """
        transformer_load = self.gridMgr.get_transformer_loading()
        line_load = self.gridMgr.get_bus_line_loading(agent)

        transformer_load = self._normalize(transformer_load, 0, 100)
        line_load = self._normalize(line_load, 0, 100)

        total_loading_cost = 0
        for index, load in enumerate(transformer_load):
            total_loading_cost += load
        for index, load in enumerate(line_load):
            total_loading_cost += load

        total_loading_cost = total_loading_cost / 6

        return -(total_loading_cost * self.load_cost_scaling)

    def _get_ev_flexibility_reward(self, agent):
        """
        Returns a reward based on the charging state of the EVs, i.e. the flexibility margin.
        """
        vehicles_present = self.gridMgr.get_bus_ev_present(agent)
        ev_soc = self.gridMgr.get_bus_ev_soc(agent)
        total_flexibility_margin = 0
        for i, ev in enumerate(vehicles_present):
            if ev:
                flexibility_margin = ev_soc[i] - self.ev_target
                total_flexibility_margin += flexibility_margin

        return total_flexibility_margin * self.ev_flexibility_scaling

    def _get_voltage_reward(self, bus):
        """
        Returns a cost based on the voltage deviation from the nominal value.
        """
        voltage = self.gridMgr.get_bus_voltage_pu()

        voltage = 1 - voltage

        voltage_cost = sum([val**2 for val in voltage])

        voltage_cost = self._normalize(voltage_cost, 0, 0.01)

        return -(voltage_cost * self.voltage_cost_scaling)

    def _get_energy_loss_reward(self):
        """
        Returns a cost based on the energy losses in the grid. (Not used in the current use cases)
        """
        line_losses = self.gridMgr.get_line_losses()
        transformer_losses = self.gridMgr.get_transformer_losses()
        energy_loss = sum(line_losses) + sum(transformer_losses)
        return energy_loss * self.energy_loss_cost_scaling

    def _get_pv_curtailment_cost(self, agent):
        """
        Returns a cost based on the curtailment of PV power.
        """
        pv_curtailment = self.gridMgr.get_pv_curtailment()
        pv_curtailment = sum(pv_curtailment)

        pv_curtailment = self._normalize(pv_curtailment, 0, 0.01)

        return -(pv_curtailment * self.curtailment_cost_scaling)

    def _take_action(self, action):
        """
        Takes an action in the environment and returns the penalties.
        Provides the random numbers for each stochastic component of the environment.
        """
        random_numbers = self.np_random.random((self.gridMgr.get_controllers_count()[2], 3))

        aborted, penalties = self.gridMgr.step_multiagent(actions=action, timestep=self.t, random_numbers=random_numbers)
        return aborted, penalties

    def _check_if_done(self):
        """
        Returns True if the episode is done.
        """
        done = self.t == (self.start + (self.num_days * 96))
        return done

    def _get_info(self, penalties, rewards):
        """
        Returns a dictionary with further information about the environment state.
        """
        load_power, pv_power, storage_power, ev_power, hp_power = self.gridMgr.get_current_power()
        hp_storage = self.gridMgr.get_heat_storage()
        pv_q_power = self.gridMgr.get_pv_reactive_power()
        prev_hp_power = self.gridMgr.get_prev_heat_pump_power()
        residual_power = pv_power.sum() - load_power.sum()
        soc_percent = self.gridMgr.get_storage_soc()
        ev_soc_percent = self.gridMgr.get_ev_soc()
        ev_present = self.gridMgr.get_ev_present()
        line_loading = self.gridMgr.get_line_loading()
        trafo_loading = self.gridMgr.get_transformer_loading()
        voltage = self.gridMgr.get_voltage_pu()
        ext_power = self.gridMgr.get_ext_power()
        pv_curtailment = self.gridMgr.get_pv_curtailment()
        price = self._current_price()

        rewards = {"reward_" + key: val for key, val in rewards.items()}
        return {
            "timestep": self.t,
            "pv_power": pv_power,
            "pv_q_power": pv_q_power,
            "pv_curtailment": pv_curtailment,
            "load_power": load_power,
            "storage_power": storage_power,
            "ev_power": ev_power,
            "hp_power": hp_power,
            "hp_storage": hp_storage,
            "prev_hp_power": prev_hp_power,
            "residual_power": residual_power,
            "soc_percent": soc_percent,
            "ev_soc_percent": ev_soc_percent,
            "ev_present": ev_present,
            "line_loading": line_loading,
            "trafo_loading": trafo_loading,
            "voltage": voltage,
            "ext_power": ext_power,
            "price": price,
            "battery_penalty": penalties["battery_penalty"],
            "charger_penalty": penalties["charger_penalty"],
            "pv_penalty": penalties["pv_penalty"],
            "hp_penalty": penalties["hp_penalty"],
            **rewards,
        }

    def _current_price_forecast(self, horizon=1, normalize=False):
        """
        Returns the price forecast for the next horizon timesteps.
        """
        if normalize:
            price = np.array(self.prices["price_clipped_normalized"].iloc[self.t : self.t + horizon].values)
        else:
            price = np.array(self.prices["price_clipped"].iloc[self.t : self.t + horizon].values) * 10
        return price.flatten()

    def _current_price(self, normalize=False):
        """
        Returns the current price.
        """

        if normalize:
            price = self.prices["price_clipped_normalized"].iloc[self.t]
        else:
            price = self.prices["price_clipped"].iloc[self.t] * 10
        return price

    def _normalize(self, value, min_val, max_val):
        """
        Normalizes the value between min_val and max_val using min/max normalization.
        Output is between 0 and 1.
        """
        return (value - min_val) / (max_val - min_val)

    def get_controllable_storage_power(self):
        return self.gridMgr.get_controllable_storage_power()

    def get_controllable_ev_power(self):
        return self.gridMgr.get_controllable_ev_power()

    def get_controllable_hp_power(self):
        return self.gridMgr.get_controllable_hp_power()

    def get_controllers_count(self):
        """
        Returns a tuple of the number of each controller type.
        (storage_controllers, pv_controllers, ev_controllers, hp_controllers)
        """
        return self.gridMgr.get_controllers_count()

    def create_flattened_box_space(self, observation_space):
        """
        Flattens a dict observation to a box space.
        """
        total_dim = 0
        for key, space in observation_space.items():
            total_dim += np.prod(space.shape)

        low = np.concatenate([space.low.flatten() for space in observation_space.values()])
        high = np.concatenate([space.high.flatten() for space in observation_space.values()])

        return spaces.Box(low=low, high=high, shape=(total_dim,), dtype=np.float32)
