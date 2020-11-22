from ia.janggi_network import JanggiNetwork
from ia.trainer import Trainer

trainer = Trainer(JanggiNetwork(), n_simulations=400, iter_max=20, n_simulation_opponent=800)
trainer.learn_policy(n_iterations=1, n_episodes=1)