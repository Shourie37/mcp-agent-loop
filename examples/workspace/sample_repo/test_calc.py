import unittest

from calc import add, multiply


class CalcTest(unittest.TestCase):
    def test_add(self) -> None:
        self.assertEqual(add(2, 3), 5)

    def test_multiply(self) -> None:
        self.assertEqual(multiply(4, 5), 20)

    def test_add_floats(self) -> None:
        self.assertAlmostEqual(add(0.1, 0.2), 0.3)


if __name__ == "__main__":
    unittest.main()
