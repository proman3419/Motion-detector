from dataclasses import dataclass


@dataclass
class SensitivityArea:
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    min_size: int
