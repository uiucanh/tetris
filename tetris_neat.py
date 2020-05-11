import sys
import io
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

quiet = "--quiet" in sys.argv
pyboy = PyBoy('tetris_1.gb', window_type='headless' if quiet else 'SDL2',
              game_wrapper=True)
pyboy.set_emulation_speed(0)

tetris = pyboy.game_wrapper()
tetris.start_game()

# Set block animation to fall instantly
pyboy.set_memory_value(0xff9a, 0)


def eval_genomes(genomes, config):
    global max_fitness
    models = []
    population = []

    for genome_id, genome in genomes:
        genome.fitness = 0
        model = neat.nn.FeedForwardNetwork.create(genome, config)
        models.append(model)
        population.append(genome)

    child = 0
    block_pool = []
    while child < pop_size:
        print("-" * 20)
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
                score = get_score(tetris.game_area(), models[child],
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
                score = get_score(tetris.game_area(), models[child],
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
                score = get_score(tetris.game_area(), models[child],
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
            if pyboy.get_memory_value(0xffe1) == 13 or action_count > 100:
                # Giving more weights to lines score
                run_fitness = tetris.fitness + \
                              pyboy.get_memory_value(0xff9e) * 100
                child_fitnesses.append(run_fitness)
                print("Iteration %s - child %s - run %s" %
                      (epoch, child, c_run))
                print("Fitness: %s" % child_fitnesses)
                c_run += 1
                tetris.reset_game()

        population[child].fitness = np.mean(child_fitnesses)
        # Dump best model
        if population[child].fitness > max_fitness:
            max_fitness = population[child].fitness
            file_name = datetime.strftime(datetime.now(),
                                          '%d_%H_%M_') + str(
                np.round(max_fitness, 2))
            with open('neat_models/%s' % file_name, 'wb') as f:
                pickle.dump(models[child], f)
        child += 1


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(
        neat.Checkpointer(5, filename_prefix='checkpoint/neat-checkpoint-'))

    winner = p.run(eval_genomes, 25)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    node_names = {-1: 'agg_height', -2: 'e_lines',
                  -3: 'n_holes', -4: 'bumpbiness', 0: 'Score'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config', 'config-feedforward.txt')
    run(config_path)
