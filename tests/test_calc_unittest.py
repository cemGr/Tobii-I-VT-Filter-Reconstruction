# tests/test_calc_unittest.py
import unittest
from app.calc import add, divide

class TestCalc(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n>> setUpClass läuft einmal vor allen Tests")

    def setUp(self):
        self.values = (10, 5)

    def test_add(self):
        a, b = self.values
        self.assertEqual(add(a, b), 15)

    def test_divide(self):
        a, b = self.values
        self.assertEqual(divide(a, b), 2)

    def test_divide_zero(self):
        a, b = self.values
        with self.assertRaises(ZeroDivisionError):
            divide(a, 0)

    def tearDown(self):
        print(">> tearDown nach jedem Test")

    @classmethod
    def tearDownClass(cls):
        print(">> tearDownClass am Ende aller Tests")

if __name__ == "__main__":
    unittest.main()
