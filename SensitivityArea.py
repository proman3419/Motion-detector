from dataclasses import dataclass


@dataclass
class SensitivityArea:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    min_size: float
