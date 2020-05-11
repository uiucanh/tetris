import sys
import io
import multiprocessing
import neat
import numpy as np
import os
import pickle
import visualize

from datetime import datetime
from core.neat import get_score
from core.utils import check_needed_turn, do_action, drop_down, \
    do_sideway, do_turn, check_needed_dirs
from pyboy import PyBoy

max_fitness = 0
epoch = 0
start_y = 24
blank_tile = 47
pop_size = 50
run_per_child = 3


def eval_genome(genome, config):
    global max_fitness

    pyboy = PyBoy('tetris_1.gb', window_type='quiet',
                  game_wrapper=True)
    pyboy.set_emulation_speed(0)

    tetris = pyboy.game_wrapper()
    tetris.start_game()

    # Set block animation to fall instantly
    pyboy.set_memory_value(0xff9a, 0)

    model = neat.nn.FeedForwardNetwork.create(genome, config)
    block_pool = []
    c_run = 0
    child_fitnesses = []
    action_count = 0

    while c_run < run_per_child:
        block_pool = [0, 4, 8, 12, 16, 20, 24] if len(block_pool) == 0 \
            else block_pool
        next_piece = np.random.choice(block_pool)
        block_pool.remove(next_piece)
        # Set the next piece to random
        pyboy.set_memory_value(0xc213, next_piece)

        # Beginning of action
        best_action_score = -np.inf
        best_action = {'Turn': 0, 'Left': 0, 'Right': 0}
        begin_state = io.BytesIO()
        begin_state.seek(0)
        beginning = pyboy.save_state(begin_state)

        # Determine how many possible rotations
        # we need to check for the block
        block_tile = pyboy.get_memory_value(0xc203)
        turns_needed = check_needed_turn(block_tile)
        lefts_needed, rights_needed = check_needed_dirs(block_tile)

        # Do middle
        for move_dir in do_action('Middle', n_dir=1,
                                  n_turn=turns_needed, pyboy=pyboy):
            score = get_score(tetris.game_area(), model,
                              pyboy)
            if score is not None and score > best_action_score:
                best_action_score = score
                best_action = {'Turn': move_dir['Turn'],
                               'Left': move_dir['Left'],
                               'Right': move_dir['Right']}
            begin_state.seek(0)
            pyboy.load_state(begin_state)

        # Do left
        for move_dir in do_action('Left', n_dir=lefts_needed,
                                  n_turn=turns_needed, pyboy=pyboy):
            score = get_score(tetris.game_area(), model,
                              pyboy)
            if score is not None and score > best_action_score:
                best_action_score = score
                best_action = {'Turn': move_dir['Turn'],
                               'Left': move_dir['Left'],
                               'Right': move_dir['Right']}
            begin_state.seek(0)
            pyboy.load_state(begin_state)

        # Do right
        for move_dir in do_action('Right', n_dir=rights_needed,
                                  n_turn=turns_needed, pyboy=pyboy):
            score = get_score(tetris.game_area(), model,
                              pyboy)
            if score is not None and score > best_action_score:
                best_action_score = score
                best_action = {'Turn': move_dir['Turn'],
                               'Left': move_dir['Left'],
                               'Right': move_dir['Right']}
            begin_state.seek(0)
            pyboy.load_state(begin_state)

        # Do best action
        for i in range(best_action['Turn']):
            do_turn(pyboy)
        for i in range(best_action['Left']):
            do_sideway(pyboy, 'Left')
        for i in range(best_action['Right']):
            do_sideway(pyboy, 'Right')
        drop_down(pyboy)
        pyboy.tick()
        action_count += 1
        best_action_score = -np.inf

        # Game over:
        if pyboy.get_memory_value(0xffe1) == 13 or action_count > 500:
            # Giving more weights to lines score
            run_fitness = pyboy.get_memory_value(0xff9e) + 1
            child_fitnesses.append(run_fitness)
            # print("Iteration %s - child %s - run %s" %
            #       (epoch, child, c_run))
            # print("Fitness: %s" % child_fitnesses)
            c_run += 1
            tetris.reset_game()

    fitness = np.mean(child_fitnesses)
    # Dump best model
    if fitness > max_fitness:
        max_fitness = fitness
        file_name = str(np.round(max_fitness, 2))
        if fitness > 100:
            with open('neat_models/%s' % file_name, 'wb') as f:
                pickle.dump(model, f)
    return fitness


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    p = neat.Checkpointer().restore_checkpoint('checkpoint/neat-checkpoint-22')
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(
        neat.Checkpointer(1, filename_prefix='checkpoint/neat-checkpoint-'))

    pe = neat.ParallelEvaluator(int(multiprocessing.cpu_count() / 2),
                                eval_genome)
    winner = p.run(pe.evaluate, 18)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    node_names = {-1: 'agg_height', -2: 'n_holes', -3: 'bumpiness',
                  -4: 'shortest', -5: 'tallest', -6: 'max_height_diff',
                  -7: 'max_col_holes', -8: 'n_col_with_holes',
                  -9: 'num_pieces', -10: 'num_pits',
                  0: 'Score'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config', 'config-feedforward.txt')
    run(config_path)
