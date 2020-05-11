import sys
import io

from datetime import datetime
from core.utils import check_needed_turn, do_action, drop_down, \
    do_sideway, do_turn, check_needed_dirs
from core.gen_algo import *
from pyboy import PyBoy, WindowEvent


max_fitness = 0
population = None
epoch = 0
pop_size = 50

quiet = "--quiet" in sys.argv
pyboy = PyBoy('tetris_1.gb', game_wrapper=True,
              window_type="headless" if quiet else "SDL2")
pyboy.set_emulation_speed(0)


tetris = pyboy.game_wrapper()
tetris.start_game()

# Set block animation to fall instantly
pyboy.set_memory_value(0xff9a, 0)
# action_count = 0
block_pool = []
while epoch < 25:
    start_time = datetime.now()
    if population is None:
        population = Population(size=pop_size)
    else:
        population = Population(size=pop_size, old_population=population)
    child = 0
    action_count = 0

    while child < pop_size:
        block_pool = [0, 4, 8, 12, 16, 20, 24] if len(block_pool) == 0 else \
            block_pool
        next_piece = np.random.choice(block_pool)
        block_pool.remove(next_piece)
        # Set the next piece to random
        pyboy.set_memory_value(0xc213, next_piece)

        # Beginning of action
        best_child_score = -np.inf
        best_action = {'Turn': 0, 'Left': 0, 'Right': 0}
        begin_state = io.BytesIO()
        begin_state.seek(0)
        beginning = pyboy.save_state(begin_state)

        # Determine how many possible rotations we need to check for the block
        block_tile = pyboy.get_memory_value(0xc203)
        turns_needed = check_needed_turn(block_tile)
        lefts_needed, rights_needed = check_needed_dirs(block_tile)

        # Do middle
        for move_dir in do_action('Middle', pyboy, n_dir=1,
                                  n_turn=turns_needed):
            score = get_score(tetris.game_area(), population.models[child],
                              pyboy)
            if score is not None and score > best_child_score:
                best_child_score = score
                best_action = {'Turn': move_dir['Turn'],
                               'Left': move_dir['Left'],
                               'Right': move_dir['Right']}
            begin_state.seek(0)
            pyboy.load_state(begin_state)

        # Do left
        for move_dir in do_action('Left', pyboy, n_dir=lefts_needed,
                                  n_turn=turns_needed):
            score = get_score(tetris.game_area(), population.models[child],
                              pyboy)
            if score is not None and score > best_child_score:
                best_child_score = score
                best_action = {'Turn': move_dir['Turn'],
                               'Left': move_dir['Left'],
                               'Right': move_dir['Right']}
            begin_state.seek(0)
            pyboy.load_state(begin_state)

        # Do right
        for move_dir in do_action('Right', pyboy, n_dir=rights_needed,
                                  n_turn=turns_needed):
            score = get_score(tetris.game_area(), population.models[child],
                              pyboy)
            if score is not None and score > best_child_score:
                best_child_score = score
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
        best_child_score = -np.inf
        action_count += 1

        # Game over:
        if pyboy.get_memory_value(0xffe1) == 13 or action_count > 100:
            # Giving more weights to lines score
            child_fitness = tetris.fitness + pyboy.get_memory_value(0xff9e) \
                            * 100
            population.fitnesses = np.append(population.fitnesses,
                                             child_fitness)
            print("-" * 20)
            print("Iteration %s - child %s" % (epoch, child))
            print("Fitness: %s" % population.fitnesses[-1])
            child += 1
            tetris.reset_game()

    print("Iteration %s fitnesses %s" % (epoch, population.fitnesses))
    print(
        "Iteration %s max fitness %s " % (epoch, np.max(population.fitnesses)))
    print(
        "Iteration %s mean fitness %s " % (epoch, np.mean(
            population.fitnesses)))
    print("Time took", datetime.now() - start_time)

    if np.max(population.fitnesses) > max_fitness:
        max_fitness = np.max(population.fitnesses)
        file_name = datetime.strftime(datetime.now(), '%d_%H_%M_') + str(
            np.round(max_fitness, 2))
        torch.save(
            population.models[np.argmax(population.fitnesses)].state_dict(),
            'models/%s' % file_name)
    epoch += 1

pyboy.stop()
