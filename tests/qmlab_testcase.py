from abc import ABC

import pytest
import unittest
import time


class QMLabTestCase(unittest.TestCase, ABC):

    def setUp(self) -> None:
        self._start = time.time()
        self._class_location = __file__

    def tearDown(self):
        time_elapsed = time.time() - self._start
        if time_elapsed > 10.0:
            print(f"({round(time_elapsed, 2):.2f} secondes)", flush=True)
