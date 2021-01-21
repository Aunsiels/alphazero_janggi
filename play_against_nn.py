from ia.predictors import FilePredictor
from ia.random_mcts_player import NNPlayer, fight
from janggi.human_player import HumanPlayer
from janggi.parameters import N_ITERATIONS, DEFAULT_N_SIMULATIONS
from janggi.utils import Color

# Example of command:
#    python3 play_against_nn.py --number_simulations 800 --n_iterations 200 --root_file_inference /tmp/showmatch --parallel_mcts True --n_threads_mcts 10

player_blue = HumanPlayer(Color.BLUE)

player_red = NNPlayer(Color.RED, n_simulations=DEFAULT_N_SIMULATIONS,
                      janggi_net=FilePredictor(),
                      temperature_start=0.01,
                      temperature_threshold=30,
                      temperature_end=0.01,
                      think_when_other=True, print_info=True)

fight(player_blue, player_red, N_ITERATIONS, print_board=True)
