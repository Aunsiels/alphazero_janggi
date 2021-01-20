from ia.predictors import FilePredictor
from ia.utils import generate_games
from janggi.parameters import N_ITERATIONS, DEFAULT_N_SIMULATIONS, N_PROCESSUS, N_EPISODES


# Example of command:
#    python3 game_generation_files.py --root_file_inference /tmp --n_iterations 200 --number_simulations 800 --n_processus 64 --n_episodes 128

WITH_POOL = False

if __name__ == "__main__":
    predictor = FilePredictor()
    generate_games(predictor, DEFAULT_N_SIMULATIONS, N_ITERATIONS, WITH_POOL, N_PROCESSUS, N_EPISODES)
