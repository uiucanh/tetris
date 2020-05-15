import io
import neat
import numpy as np
import os
import pickle
import visualize

from core.gen_algo import get_score
from core.utils import check_needed_turn, do_action, drop_down, \
    do_sideway, do_turn, check_needed_dirs
from multiprocessing import cpu_count
from pyboy import PyBoy, WindowEvent


epochs = 10
max_fitness = 0
blank_tile = 47
pop_size = 50
max_action_count = 1000
max_score = 999999
n_workers = int(cpu_count() / 2)


def custom_start(pyboy):
    # This replaces pyboy start function to get some randomness
    while True:
        pyboy.tick()
        pyboy.botsupport_manager().tilemap_background().refresh_lcdc()
        if pyboy.botsupport_manager().tilemap_background()[2:9, 14] == \
                [89, 25, 21, 10, 34, 14, 27]:
            break

    np.random.seed()
    for _ in range(np.random.randint(1, 60, size=(1,))[0]):
        pyboy.tick()

    for _ in range(3):
        pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
        pyboy.tick()
        pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)

        for _ in range(6):
            pyboy.tick()


def eval_genome(genome, config):
    global max_fitness

    pyboy = PyBoy('tetris.gb', window_type='quiet',
                  game_wrapper=True)
    pyboy.set_emulation_speed(0)
    tetris = pyboy.game_wrapper()
    custom_start(pyboy)

    # Set block animation to fall instantly
    pyboy.set_memory_value(0xff9a, 1)

    model = neat.nn.FeedForwardNetwork.create(genome, config)
    child_fitness = 0
    action_count = 0

    while not pyboy.tick():
        # Beginning of action
        best_action_score = np.NINF
        best_action = {'Turn': 0, 'Left': 0, 'Right': 0}
        begin_state = io.BytesIO()
        begin_state.seek(0)
        pyboy.save_state(begin_state)

        # Determine how many possible rotations
        # we need to check for the block
        block_tile = pyboy.get_memory_value(0xc203)
        turns_needed = check_needed_turn(block_tile)
        lefts_needed, rights_needed = check_needed_dirs(block_tile)

        # Do middle
        for move_dir in do_action('Middle', n_dir=1,
                                  n_turn=turns_needed, pyboy=pyboy):
            score = get_score(tetris.game_area(), model,
                              pyboy, neat=True)
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
                              pyboy, neat=True)
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
                              pyboy, neat=True)
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

        # Game over:
        if pyboy.get_memory_value(0xffe1) == 13 or \
                tetris.score == max_score:
            child_fitness = tetris.score
            if tetris.score == max_score:
                print("Max score reached")
            break

    # Dump best model
    if child_fitness > max_fitness:
        max_fitness = child_fitness
        file_name = str(np.round(max_fitness, 2))
        if tetris.level >= 20:
            with open('neat_models/%s' % file_name, 'wb') as f:
                pickle.dump(model, f)
    pyboy.stop()
    return child_fitness


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    # p = neat.Population(config)
    p = neat.Checkpointer().restore_checkpoint('checkpoint/neat-checkpoint-8')
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(
        neat.Checkpointer(1, filename_prefix='checkpoint/neat-checkpoint-'))

    pe = neat.ParallelEvaluator(n_workers, eval_genome)
    winner = p.run(pe.evaluate, 20)

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
