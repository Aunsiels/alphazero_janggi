import multiprocessing as mp
import time

from ia.trainer import ModelSaver, run_episode_raw, run_episode_raw_loop


def generate_games(predictor, n_simulations, iter_max, with_pool, n_processes, n_episodes):
    model_saver = ModelSaver()
    while True:
        if with_pool:
            begin_time = time.time()
            with mp.Pool(n_processes) as pool:
                episodes = pool.map(run_episode_raw,
                                    [(predictor, n_simulations, iter_max) for _ in range(n_episodes)])
            model_saver.save_episodes_raw(episodes)
            print("Total time:", time.time() - begin_time)
        else:
            output_queue = mp.Queue()
            processes = []
            saving_process = mp.Process(target=save_queue_process,
                                        args=(output_queue, n_episodes))
            saving_process.start()
            for _ in range(n_processes):
                processes.append(mp.Process(target=run_episode_raw_loop,
                                            args=(predictor, n_simulations, iter_max, output_queue)))
            for process in processes:
                process.start()
            for process in processes:
                process.join()


def save_queue_process(queue, n_episodes):
    model_saver = ModelSaver()
    begin_time = time.time()
    while True:
        if queue.qsize() < n_episodes:
            time.sleep(1)
            continue
        episodes = []
        for _ in range(n_episodes):
            episodes.append(queue.get())
        model_saver.save_episodes_raw(episodes)
        print("Total time:", time.time() - begin_time)
        begin_time = time.time()