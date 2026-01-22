import pickle
import numpy as np
import argparse
import visualize
import os
import neat

from core.utils import do_best_action, spawn_pyboy
from core.gen_algo import get_score


def main(model_path, runs, draw, config_path):

    pyboy, tetris = spawn_pyboy(show=True)

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    scores = []
    lines = []
    n = 0

    while n < runs:

        do_best_action(get_score, pyboy, tetris, model, neat=True)

        if tetris.game_over():
            print(tetris.score)
            print(tetris.lines)
            scores.append(tetris.score)
            lines.append(tetris.lines)
            n += 1
            tetris.reset_game()

    print("Scores:", scores)
    print("Average:", np.average(scores))
    print("---")
    print("Lines:", lines)
    print("Average:", np.average(lines))

    if draw:
        config = neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
        )
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

        with open(model_path + '_genome', 'rb') as f:
            genome = pickle.load(f)
        visualize.draw_net(config, genome, True, node_names=node_names)


def cli_args():
    parser = argparse.ArgumentParser("Tetris neat model player")
    parser.add_argument(
        'model_path',
        help="Path to neat model",
        type=str,
    )
    parser.add_argument(
        '-r',
        '--runs',
        help="Tetris games played",
        type=int,
        default=1,
    )
    parser.add_argument(
        '-d',
        '--draw',
        help="Draw network after playing",
        default=False,
        action='store_true',
    )

    return parser.parse_args()


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config', 'config-feedforward.txt')

    args = cli_args()
    main(
        model_path=args.model_path,
        runs=args.runs,
        draw=args.draw,
        config_path=config_path,
    )
