import argparse

parser = argparse.ArgumentParser(description='A program to train Janggi IA.')
parser.add_argument("--stockfish_location",
                    default="D:/Downloads/fairy-stockfish-largeboard_x86-64.exe", type=str,
                    required=False, help="The location of the game engine.")
parser.add_argument("--max_repetitions",
                    default=3, type=int,
                    required=False, help="The number of times a board position can repeat.")
parser.add_argument("--n_residuals",
                    default=2, type=int,
                    required=False, help="The number of residual layer in the neural network.")
parser.add_argument("--dirichlet_alpha",
                    default=0.03, type=float,
                    required=False, help="Dirichlet alpha used at the beginning of the MCTS.")
parser.add_argument("--dirichlet_epsilon",
                    default=0.25, type=float,
                    required=False, help="Proportion of dirichlet distribution at the beginning of the MCTS.")
parser.add_argument("--temperature_start",
                    default=1.0, type=float,
                    required=False, help="Temperature at the beginning of the MCTS.")
parser.add_argument("--temperature_end",
                    default=1.0, type=float,
                    required=False, help="Temperature at the end of the MCTS.")
parser.add_argument("--temperature_threshold",
                    default=30, type=int,
                    required=False, help="Round threshold at which we change the temperature.")
parser.add_argument("--number_simulations",
                    default=800, type=int,
                    required=False, help="Number of simulations in the MCTS.")
parser.add_argument("--c_puct",
                    default=4.0, type=float,
                    required=False, help="C_PUCT parameter in the MCTS.")

parser.add_argument("--prop_population_learning",
                    default=1/50, type=float,
                    required=False, help="Proportion of the moves to consider during training.")
parser.add_argument("--n_last_games",
                    default=500000, type=int,
                    required=False, help="Number of previous games to consider for training.")
parser.add_argument("--learning_rate",
                    default=0.001, type=float,
                    required=False, help="Learning rate during training.")
parser.add_argument("--n_epoch",
                    default=1, type=int,
                    required=False, help="Number of epoch during training.")
parser.add_argument("--n_epoch_continuous",
                    default=1, type=int,
                    required=False, help="Number of times to train between two evaluations in continuous learning.")
parser.add_argument("--waiting_time_no_episode",
                    default=1000, type=float,
                    required=False, help="Waiting time if there is no episode to consider (in seconds).")
parser.add_argument("--n_fights",
                    default=100, type=int,
                    required=False, help="Number of fights in evaluation.")
parser.add_argument("--victory_threshold",
                    default=55.0, type=float,
                    required=False, help="Percentage of victory to update model.")
parser.add_argument("--n_games_supervised",
                    default=30000, type=int,
                    required=False, help="Number of games to consider at once in supervised learning.")
parser.add_argument("--log_frequency",
                    default=1000, type=int,
                    required=False, help="How often should we write log during training.")
parser.add_argument("--batch_size",
                    default=16, type=int,
                    required=False, help="Batch size during the training.")


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser.add_argument("--parallel_mcts", default=False, type=str2bool, required=False,
                    help="Whether or not to run a parallel MCTS")
parser.add_argument("--n_threads_mcts", default=1, type=int, required=False,
                    help="Number of threads when running MCTS in parallel. Needs --parallel_mcts True")

parser.add_argument("--n_processus", default=1, type=int, required=False,
                    help="Number of processus")

parser.add_argument("--root_file_inference", default="./", type=str, required=False,
                    help="Root directory for the file inference.")

parser.add_argument("--n_iterations", default=100, type=int, required=False,
                    help="The number of iterations")
parser.add_argument("--n_episodes", default=2, type=int, required=False,
                    help="Number of episodes.")

parser.add_argument("--train_on_all", default=False, type=str2bool, required=False,
                    help="Train on all generated games, not only the n_last_games. Done by chunks of n_last_games.")

parser.add_argument("--train_new_model", default=False, type=str2bool, required=False,
                    help="Train a new model from scratch.")

args = parser.parse_args()

STOCKFISH_LOCATION = args.stockfish_location  # 'D:/Downloads/fairy-stockfish-largeboard_x86-64.exe'
MAX_REPETITIONS = args.max_repetitions
N_RESIDUAL_DEFAULT = args.n_residuals

DIRICHLET_ALPHA = args.dirichlet_alpha
DIRICHLET_EPSILON = args.dirichlet_epsilon

DEFAULT_TEMPERATURE_START = args.temperature_start
DEFAULT_TEMPERATURE_END = args.temperature_end
DEFAULT_TEMPERATURE_THRESHOLD = args.temperature_threshold

DEFAULT_N_SIMULATIONS = args.number_simulations
DEFAULT_C_PUCT = args.c_puct

# Learning rate of the optimizer
PROP_POPULATION_FOR_LEARNING = args.prop_population_learning
N_LAST_GAME_TO_CONSIDER = args.n_last_games
LEARNING_RATE = args.learning_rate

# Number of epoch when learning
EPOCH_NUMBER = args.n_epoch
# Number of epoch when learning continuously
EPOCH_NUMBER_CONTINUOUS = args.n_epoch_continuous
# When there is no episode to process, just wait
WAINTING_TIME_IF_NO_EPISODE = args.waiting_time_no_episode

# For check if a model is better than the previous one, we perform some fights
N_FIGHTS = args.n_fights
# If the new model wins more than a certain percentage of games, we update the current model
VICTORY_THRESHOLD = args.victory_threshold

# When learning with existing data, how many game do we consider at once
SUPERVISED_GAMES_FREQ = args.n_games_supervised

# During training, how often do we print loss
LOG_PRINT_FREQ = args.log_frequency

# For training
BATCH_SIZE = args.batch_size
PARALLEL_MCTS = args.parallel_mcts
N_THREADS_MCTS = args.n_threads_mcts

BASE_ROOT_FILES = args.root_file_inference
N_ITERATIONS = args.n_iterations
N_PROCESSUS = args.n_processus
N_EPISODES = args.n_episodes

TRAIN_ON_ALL = args.train_on_all
TRAIN_NEW_MODEL = args.train_new_model
