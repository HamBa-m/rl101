# =========================
# models.py (SIMPLIFIED & DETERMINISTIC)
# =========================

import numpy as np

PANEL_COUNT = 100
PANEL_RATED_POWER_KW = 0.55
PERFORMANCE_RATIO = 0.78


def _solar_peak_output_kw() -> float:
    return PANEL_COUNT * PANEL_RATED_POWER_KW * PERFORMANCE_RATIO


def get_solar_output(hour: int) -> float:
    """Deterministic solar generation (no noise)."""
    if 6 <= hour <= 18:
        val = _solar_peak_output_kw() * np.sin(np.pi * (hour - 6) / 12)
        return max(0.0, val)
    return 0.0


def get_demand_profile(hour: int) -> float:
    """Deterministic demand profile."""
    morning_peak = 5.0 * np.exp(-((hour - 8) ** 2) / 6.0)
    midday_peak = 9.0 * np.exp(-((hour - 12) ** 2) / 10.0)
    evening_peak = 6.0 * np.exp(-((hour - 20) ** 2) / 12.0)
    base = 3.0
    return max(2.0, base + morning_peak + midday_peak + evening_peak)


def get_tariff(hour: int) -> float:
    """Simple peak/off-peak tariff."""
    return 21.0 if 18 <= hour <= 22 else 18.0