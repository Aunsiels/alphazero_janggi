import unittest

from ia.janggi_network import JanggiNetwork
from ia.random_mcts_player import fight, NNPlayer
from ia.trainer import Trainer
from janggi.utils import Color


class TestTrainer(unittest.TestCase):

    def test_first(self):
        trainer = Trainer(JanggiNetwork(), 10, 10)
        examples = trainer.run_episode()
        self.assertEqual(len(examples), 20)
        for example in examples:
            self.assertIn(example[2], [-1, 1])

    def test_learn(self):
        trainer = Trainer(JanggiNetwork(), n_simulations=100, iter_max=30, n_simulation_opponent=10)
        trainer.learn_policy(n_iterations=1, n_episodes=10)

    def test_fight(self):
        trainer = Trainer(JanggiNetwork(), n_simulations=10, iter_max=30, n_simulation_opponent=10)
        trainer.train_and_fight([])

    def test_fight2(self):
        player_blue = NNPlayer(Color.BLUE, n_simulations=100,
                               janggi_net=JanggiNetwork(),
                               temperature_start=0.01,
                               temperature_threshold=30,
                               temperature_end=0.01)
        player_red = NNPlayer(Color.RED, n_simulations=100,
                               janggi_net=JanggiNetwork(),
                               temperature_start=0.01,
                               temperature_threshold=30,
                               temperature_end=0.01)
        fight(player_blue, player_red, 100)


if __name__ == '__main__':
    unittest.main()
