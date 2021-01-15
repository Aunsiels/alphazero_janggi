import time
import multiprocessing as mp

from ia.trainer import run_episode_raw_not_nn, ModelSaver, run_episode_stockfish

N_POOLS = 2
N_SIMULATIONS = 1600
ITER_MAX = 200
N_EPISODES = N_POOLS * 1
STOCKFISH = True


if __name__ == "__main__":
    while True:
        begin_time = time.time()
        with mp.Pool(N_POOLS) as pool:
            if STOCKFISH:
                episodes = pool.map(run_episode_stockfish,
                                    [ITER_MAX for _ in range(N_EPISODES)])
            else:
                episodes = pool.map(run_episode_raw_not_nn,
                                    [(N_SIMULATIONS, ITER_MAX) for _ in range(N_EPISODES)])
        model_saver = ModelSaver()
        model_saver.save_episodes_raw(episodes, mini=True)
        print("Total time:", time.time() - begin_time)
