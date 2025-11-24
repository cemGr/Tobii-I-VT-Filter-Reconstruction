import pytest

from app.calc import add, divide, mult


class TestCalc:
    def test_addiere_zwei_zahlen(self):
        assert add(2, 3) == 5

    def test_divide_ganzzahlig(self):
        assert divide(9, 2) == 4.5

    def test_division_durch_null(self):
        with pytest.raises(ZeroDivisionError):
            divide(10, 0)

    def test_mult_two_numbers(self):
        assert mult(2, 4) == 8
