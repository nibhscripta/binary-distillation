import unittest
from unittest import TestCase

from numpy import linspace

from pystill.equilibrium import EquilibriumLine
from pystill.design import DistillationColumn

eq_non_azeo = EquilibriumLine.from_function(lambda x: x**0.5)
eq_azeo = EquilibriumLine([0, 0.25, 0.5, 0.75, 1], [0, 0.4, 0.5, 0.6, 1])

class TestEquilibriumLine(TestCase):

    def test_invalid_composition(self):
        with self.assertRaises(TypeError):
            EquilibriumLine([-1], [1])
            EquilibriumLine([2], [1])
            EquilibriumLine([1], [-1])
            EquilibriumLine([1], [2])

    def test_azeo(self):
        x = [0, 0.5, 1]
        y = [0, 0.5, 1]

        self.assertEqual(EquilibriumLine(x, y).azeo_x, 0.5)

    def test_from_function(self):
        f = lambda x: x**0.5

        x = linspace(0, 1, 1000)
        y = f(x)

        self.assertEqual(EquilibriumLine.from_function(f).y.all(), y.all())

class TestDistillationColumn(TestCase):

    def test_invalid_specification(self):
        with self.assertRaises(TypeError):
            # Non azeo
            DistillationColumn(0.5, 0.4, 0.1, 1, eq_non_azeo) # x_D < x_F
            DistillationColumn(0.5, 0.9, 0.6, 1, eq_non_azeo) # x_W > x_F
            DistillationColumn(-0.5, 0.9, 0.1, 1, eq_non_azeo) # x_F < 0
            DistillationColumn(1.5, 0.9, 0.1, 1, eq_non_azeo) # x_F > 1
            DistillationColumn(0.5, -0.9, 0.1, 1, eq_non_azeo) # x_D < 0
            DistillationColumn(0.5, 1.9, 0.1, 1, eq_non_azeo) # x_D > 1
            DistillationColumn(0.5, 0.9, -0.1, 1, eq_non_azeo) # x_W < 0
            DistillationColumn(0.5, 0.9, 1.1, 1, eq_non_azeo) # x_W > 1
            # With invalid azeo spec
            DistillationColumn(0.3, 0.6, 0.1, 1, eq_non_azeo) # below azeo
            DistillationColumn(0.55, 0.6, 0.1, 1, eq_non_azeo) # below azeo
            DistillationColumn(0.6, 0.7, 0.1, 1, eq_non_azeo) # above azeo
            DistillationColumn(0.4, 0.7, 0.1, 1, eq_non_azeo) # above azeo

    def test_infeasible(self):
        with self.assertRaises(TypeError):
            DistillationColumn(0.5, 0.99, 0.01, 0.5, eq_non_azeo, R=1.37)

    def test_too_low_E(self):
        op = DistillationColumn(0.5, 0.99, 0.01, 1, eq_non_azeo, R=1.37)
        with self.assertWarns(Warning):
            op.design_stages(E=0.01, B_E=0.01)



if __name__ == '__main__':
    unittest.main()