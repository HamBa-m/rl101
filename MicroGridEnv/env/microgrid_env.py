# =========================
# microgrid_env.py (TEACHING VERSION)
# =========================

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Optional, Tuple

from .models import get_solar_output, get_demand_profile, get_tariff


class MicrogridLawEnv(gym.Env):
    """
    Simplified educational environment aligned with Morocco Law 82-21.

    Key features:
    - Deterministic dynamics
    - Daily export cap (20%)
    - Student-controlled observation space
    - Modular reward shaping
    """

    def __init__(
        self,
        reward_weights: Dict[str, float],
        selected_features: List[str],
    ):
        super().__init__()

        self.reward_weights = reward_weights
        self.selected_features = selected_features

        # Action space: simplified
        self.action_space = spaces.Discrete(3)
        # 0: Idle, 1: Charge, 2: Discharge

        self.feature_names = ["soc", "solar", "demand", "price", "hour"]

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(selected_features),),
            dtype=np.float32,
        )

        # Battery parameters
        self.max_soc = 100.0
        self.max_rate = 20.0

        self.reset()

    def _get_obs(self):
        return np.array([self.feature_map[f] for f in self.selected_features], dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        h = self.current_hour % 24

        solar = get_solar_output(h)
        demand = get_demand_profile(h)
        price = get_tariff(h)

        self.daily_gen += solar

        net_solar = max(0.0, solar - demand)
        unmet_demand = max(0.0, demand - solar)

        charge_amount = 0.0
        discharge_amount = 0.0
        bought = 0.0
        sold = 0.0

        # Daily export cap (20%)
        quota_limit = 0.2 * max(1e-5, self.daily_gen)
        quota_remaining = max(0.0, quota_limit - self.daily_sold)

        # === ACTION LOGIC ===

        if action == 0:  # Idle
            sold = min(net_solar, quota_remaining)
            bought = unmet_demand

        elif action == 1:  # Charge battery
            charge = min(net_solar + unmet_demand, self.max_soc - self.soc, self.max_rate)
            self.soc += charge
            charge_amount = charge

            bought = max(0.0, unmet_demand + charge - net_solar)
            sold = min(max(0.0, net_solar - charge), quota_remaining)

        elif action == 2:  # Discharge battery
            discharge = min(self.soc, unmet_demand + net_solar, self.max_rate)
            self.soc -= discharge
            discharge_amount = discharge

            used_for_demand = min(discharge, unmet_demand)
            remaining = discharge - used_for_demand

            bought = max(0.0, unmet_demand - used_for_demand)
            sold = min(net_solar + remaining, quota_remaining)

        self.daily_sold += sold

        # === REWARD TERMS ===

        net_profit = (sold * price) - (bought * price)
        battery_usage = charge_amount + discharge_amount

        reward_terms = {
            "profit": net_profit,
            "battery_use": -battery_usage,
            "grid_penalty": -bought,
        }

        reward = sum(
            self.reward_weights.get(k, 0.0) * v
            for k, v in reward_terms.items()
        )

        # === STATE ===

        self.feature_map = {
            "soc": self.soc / self.max_soc,
            "solar": solar,
            "demand": demand,
            "price": price,
            "hour": h,
        }

        self.current_hour += 1

        # Reset daily counters
        if self.current_hour % 24 == 0:
            self.daily_gen = 0.0
            self.daily_sold = 0.0

        terminated = self.current_hour >= 168

        info = {
            "profit": net_profit,
            "soc": self.soc,
            "action": action,
            "reward_terms": reward_terms,
        }

        return self._get_obs(), float(reward), terminated, False, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)

        self.current_hour = 0
        self.soc = 20.0

        self.daily_gen = 0.0
        self.daily_sold = 0.0

        # Initial state
        self.feature_map = {
            "soc": self.soc / self.max_soc,
            "solar": 0.0,
            "demand": 3.0,
            "price": 18.0,
            "hour": 0,
        }

        return self._get_obs(), {}
