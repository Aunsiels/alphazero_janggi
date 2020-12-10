import unittest

from janggi.action import Action


class TestAction(unittest.TestCase):

    def test_hash(self):
        all_actions = set()
        all_hashes = set()
        for x_from in range(10):
            for x_to in range(10):
                for y_from in range(9):
                    for y_to in range(9):
                        action0 = Action(x_from, y_from, x_to, y_to)
                        action1 = Action(x_from, y_from, x_to, y_to)
                        self.assertEqual(hash(action0), hash(action1))
                        all_actions.add(action0)
                        all_hashes.add(hash(action0))
        self.assertEqual(len(all_actions), 10 * 10 * 9 * 9)
        self.assertEqual(len(all_hashes), 10 * 10 * 9 * 9)

    def test_pass(self):
        action = Action(0, 0, 0, 0)
        self.assertTrue(action.is_pass())
        print(action.get_features())
