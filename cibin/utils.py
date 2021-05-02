"""
Utilities and helper functions.

These functions can be used to construct 2-sided 1 - alpha confidence
bounds for the average treatment effect in a randomized experiment with binary
outcomes and two treatments.
"""

from itertools import combinations
from math import comb, floor
import numpy as np
