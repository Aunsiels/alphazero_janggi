import unittest

from ia.janggi_network import JanggiNetwork
from ia.trainer import Trainer


class TestTrainer(unittest.TestCase):

    def test_first(self):
        trainer = Trainer(JanggiNetwork(), 10, 10)
        examples = trainer.run_episode()
        self.assertEqual(len(examples), 10)
        previous = None
        for example in examples:
            self.assertIn(example[2], [-1, 1])
            self.assertNotEqual(previous, example[2])
            previous = example[2]

    def test_learn(self):
        trainer = Trainer(JanggiNetwork(), n_simulations=30, iter_max=100, n_simulation_opponent=800)
        trainer.learn_policy(n_iterations=10, n_episodes=10)


if __name__ == '__main__':
    unittest.main()
