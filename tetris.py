import pickle
import numpy as np
import torch
import logging

from datetime import datetime
from core.gen_algo import get_score, Population
from core.utils import feature_names, spawn_pyboy, do_best_action
from multiprocessing import Pool, cpu_count

import argparse
import signal

logger = logging.getLogger("tetris")
logger.setLevel(logging.INFO)

fh = logging.FileHandler('logs.out')
fh.setLevel(logging.INFO)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


def eval_network(epoch, child_index, child_model, run_per_child, max_score,
                 show, action_limit):

    pyboy, tetris = spawn_pyboy(show=show)

    run = 0
    actions = 0
    scores = []
    levels = []
    lines = []

    while run < run_per_child:

        do_best_action(get_score, pyboy, tetris, child_model, neat=False)
        actions += 1

        highscore = tetris.score == max_score
        actions_depleted = actions > action_limit

        # Game over:
        if tetris.game_over() or highscore or actions_depleted:
            scores.append(tetris.score)
            levels.append(tetris.level)
            lines.append(tetris.lines)
            if run == run_per_child - 1:
                pyboy.stop()
            else:
                actions = 0
                tetris.reset_game()
            run += 1

    child_fitness = np.average(scores)
    logger.info("-" * 20)
    logger.info("Iteration %s - child %s" % (epoch, child_index))
    logger.info("Score: %s, Level: %s, Lines %s" % (scores, levels, lines))
    logger.info("Fitness: %s" % child_fitness)
    logger.info("Output weight:")
    weights = {}
    for i, j in zip(feature_names, child_model.output.weight.data.tolist()[0]):
        weights[i] = np.round(j, 3)
    logger.info(weights)

    return child_fitness


def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def main(n_workers, pop_size, epochs, run_per_child, max_score,
         show, action_limit):
    # TODO: pass optional checkpoint path to continue training from,
    # instead of manually setting epoch counter here
    e = 0

    p = Pool(n_workers, initializer=initializer)
    population = None
    max_fitness = 0

    while e < epochs:
        start_time = datetime.now()
        if population is None:
            if e == 0:
                population = Population(size=pop_size)
            else:
                with open('checkpoint/checkpoint-%s.pkl' % (e - 1), 'rb') as f:
                    population = pickle.load(f)
        else:
            population = Population(size=pop_size, old_population=population)

        result = [0] * pop_size
        for i in range(pop_size):
            result[i] = p.apply_async(
                eval_network,
                (
                    e,
                    i,
                    population.models[i],
                    run_per_child,
                    max_score,
                    show,
                    action_limit,
                )
            )

        for i in range(pop_size):
            try:
                population.fitnesses[i] = result[i].get()
            except KeyboardInterrupt:
                logger.error("KeyboardInterrupt, stopping ...")
                p.terminate()
                return

        logger.info("-" * 20)
        logger.info("Iteration %s fitnesses %s" % (
            e, np.round(population.fitnesses, 2)))
        logger.info(
            "Iteration %s max fitness %s " % (e, np.max(population.fitnesses)))
        logger.info(
            "Iteration %s mean fitness %s " % (e, np.mean(
                population.fitnesses)))
        logger.info("Time took %s" % (datetime.now() - start_time))
        logger.info("Best child output weights:")
        weights = {}
        for i, j in zip(feature_names, population.models[np.argmax(
                population.fitnesses)].output.weight.data.tolist()[0]):
            weights[i] = np.round(j, 3)
        logger.info(weights)
        # Saving population
        with open('checkpoint/checkpoint-%s.pkl' % e, 'wb') as f:
            pickle.dump(population, f)

        if np.max(population.fitnesses) >= max_fitness:
            max_fitness = np.max(population.fitnesses)
            file_name = datetime.strftime(datetime.now(), '%d_%H_%M_') + str(
                np.round(max_fitness, 2))
            # Saving best model
            torch.save(
                population.models[np.argmax(
                    population.fitnesses)].state_dict(),
                'models/%s' % file_name)
        e += 1

    p.close()
    p.join()


def cli_args():
    parser = argparse.ArgumentParser("Tetris training")
    parser.add_argument(
        '-e',
        '--epochs',
        help="How many epochs to train",
        type=int,
        default=2,
    )
    parser.add_argument(
        '-a',
        '--action_limit',
        help="Maximum number of moves to make in simulation",
        type=int,
        default=100,
    )
    parser.add_argument(
        '-r',
        '--runs',
        help="Tetris games played per child",
        type=int,
        default=3,
    )
    parser.add_argument(
        '-p',
        '--popsize',
        help="Size of population",
        type=int,
        default=50,
    )
    parser.add_argument(
        '-m',
        '--maxscore',
        help="Maximum score in one run, game is stopped if reached",
        type=int,
        default=999999,
    )
    parser.add_argument(
        '-n',
        '--numworkers',
        help="Number of workers (pyboys) to spawn",
        type=int,
        default=cpu_count(),
    )
    parser.add_argument(
        '-s',
        '--show',
        help="Do not run pyboys in headless mode, render each one in window",
        default=False,
        action='store_true',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = cli_args()
    main(
        n_workers=args.numworkers,
        pop_size=args.popsize,
        epochs=args.epochs,
        run_per_child=args.runs,
        max_score=args.maxscore,
        show=args.show,
        action_limit=args.action_limit,
    )
