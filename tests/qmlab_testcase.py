from abc import ABC

import os
import unittest
import time
import yaml
import numpy as np
from qiskit_algorithms.utils import algorithm_globals


fname = os.path.join(os.path.dirname(__file__), "config_test.yaml")
with open(fname) as f:
    file = yaml.safe_load(f)

random_state = file["random_state"]


class QMLabTest(unittest.TestCase, ABC):

    def setUp(self) -> None:
        self._start = time.time()
        self._class_location = __file__
        self.random_state = random_state
        np.random.seed(self.random_state)
        algorithm_globals.seed = random_state

    def tearDown(self):
        test_runtime = time.time() - self._start
        if test_runtime > 5.0:
            print(f"Test took {round(test_runtime):.2f}", flush=True)
