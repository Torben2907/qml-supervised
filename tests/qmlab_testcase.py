from abc import ABC

import unittest
import time


class QMLabTest(unittest.TestCase, ABC):

    def setUp(self) -> None:
        self._start = time.time()
        self._class_location = __file__

    def tearDown(self):
        test_runtime = time.time() - self._start
        if test_runtime > 5.0:
            print(f"Test took {round(test_runtime):.2f}", flush=True)
