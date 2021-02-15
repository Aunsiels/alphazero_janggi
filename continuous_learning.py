from ia.janggi_network import JanggiNetwork
from ia.trainer import Trainer

# Example:
# CUDA_VISIBLE_DEVICES=2 python3 continuous_learning.py --n_fights 30 --c_puct 1.0 --n_residuals 20
# CUDA_VISIBLE_DEVICES=0 python3 continuous_learning.py --n_fights 30 --c_puct 1.0 --n_iterations 200 --number_simulations 800 --n_residuals 40 --train_on_all True --train_new_model True

if __name__ == "__main__":
    trainer = Trainer(JanggiNetwork(40), n_simulations=800, iter_max=200, n_simulation_opponent=800)
    trainer.continuous_learning()
