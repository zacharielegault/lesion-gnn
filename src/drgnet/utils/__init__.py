from enum import Enum


class ClassWeights(str, Enum):
    UNIFORM = "uniform"
    INVERSE = "inverse"
    QUADRATIC_INVERSE = "quadratic_inverse"
    INVERSE_FREQUENCY = "inverse_frequency"
