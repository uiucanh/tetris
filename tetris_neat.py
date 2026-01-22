import neat
import numpy as np
import os
import pickle
import argparse

import visualize
from core.gen_algo import get_score
from core.utils import do_best_action, spawn_pyboy


def eval_genome(genome, config):

    args = config.custom_args
    pyboy, tetris = spawn_pyboy(show=args.show)

    # Set block animation to fall instantly
    pyboy.memory[0xff9a] = 2

    model = neat.nn.FeedForwardNetwork.create(genome, config)
    child_fitness = 0
    actions = 0

    while True:

        do_best_action(get_score, pyboy, tetris, model, neat=True)
        actions += 1

        game_over = tetris.game_over()
        actions_depleted = actions > args.action_limit

        # Game over:
        if game_over or actions_depleted:
            child_fitness = tetris.score
            # punish loosing
            if game_over:
                child_fitness = 0
            break

    # Dump good models
    if child_fitness > 10000:
        file_name = str(np.round(child_fitness, 2))
        with open('neat_models/%s' % file_name, 'wb') as f:
            pickle.dump(model, f)
        with open('neat_models/%s_genome' % file_name, 'wb') as f:
            pickle.dump(genome, f)

    pyboy.stop()
    return child_fitness


def run(config_path, args):

    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    if args.checkpoint_path:
        p = neat.Checkpointer().restore_checkpoint(args.checkpoint_path)
    else:
        p = neat.Population(config)

    config.custom_args = args

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(
        neat.Checkpointer(1, filename_prefix='checkpoint/neat-checkpoint-'))

    pe = neat.ParallelEvaluator(args.num_workers, eval_genome)
    winner = p.run(pe.evaluate, args.epochs)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))

    if args.draw:
        print('\nOutput:')
        node_names = {
            -1: 'agg_height',
            -2: 'n_holes',
            -3: 'bumpiness',
            -4: 'cleared',
            -5: 'num_pits',
            -6: 'max_wells',
            -7: 'n_cols_with_holes',
            -8: 'row_transitions',
            -9: 'col_transitions',
            -10: 'block_bit_1',
            -11: 'block_bit_2',
            -12: 'block_bit_3',
            0: 'Score',
        }
        visualize.draw_net(config, winner, True, node_names=node_names)
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)


def cli_args():
    parser = argparse.ArgumentParser("Tetris neat model trainer")
    parser.add_argument(
        '-c',
        '--checkpoint_path',
        help="Path to checkpoint to continue simulation",
        type=str,
    )
    parser.add_argument(
        '-e',
        '--epochs',
        help="Epochs to run simulation",
        type=int,
        default=10,
    )
    parser.add_argument(
        '-n',
        '--num_workers',
        help="Number of workers to spawn for simulation",
        type=int,
        default=16,
    )
    parser.add_argument(
        '-a',
        '--action_limit',
        help="Maximum number of moves to make in simulation",
        type=int,
        default=100,
    )
    parser.add_argument(
        '-s',
        '--show',
        help="Do not run pyboys in headless mode, render each one in window",
        default=False,
        action='store_true',
    )
    parser.add_argument(
        '-d',
        '--draw',
        help="Draw best network after simulation",
        default=False,
        action='store_true',
    )
    return parser.parse_args()


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config', 'config-feedforward.txt')
    args = cli_args()
    run(config_path, args)
