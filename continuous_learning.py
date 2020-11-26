from ia.janggi_network import JanggiNetwork
from ia.trainer import Trainer

if __name__ == "__main__":
    trainer = Trainer(JanggiNetwork(), n_simulations=10, iter_max=100, n_simulation_opponent=10)
    trainer.continuous_learning()
