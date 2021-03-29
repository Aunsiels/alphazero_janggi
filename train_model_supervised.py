from ia.janggi_network import JanggiNetwork
from ia.trainer import Trainer

if __name__ == "__main__":
    trainer = Trainer(JanggiNetwork(20), n_simulations=800, iter_max=200, n_simulation_opponent=800,
                      dir_base="model")
    trainer.learn_supervised("data/game_data.txt")
