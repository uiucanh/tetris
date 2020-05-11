import numpy as np
import sys

from core.ga import Population
from core.board_utils import get_board_info
from pyboy import PyBoy, WindowEvent

quiet = "--quiet" in sys.argv
pyboy = PyBoy('tetris.gb', window_type="headless" if quiet else "SDL2",
              game_wrapper=True, disable_renderer=True if quiet else False)
pyboy.set_emulation_speed(0)

tetris = pyboy.game_wrapper()
tetris.start_game()

max_fitness = 0
population = None
epoch = 0
blank_tile = 47
pop_size = 50

print("-" * 20)
print("Iteration %s - child %s" % (epoch, 0))

while epoch < 25:
    if population is None:
        population = Population(size=pop_size)
    else:
        population = Population(size=pop_size, old_population=population)
    child = 0

    output_count = {
        'left': 0,
        'right': 0,
        'down': 0,
        'a': 0
    }
    while child < pop_size:
        pyboy.set_memory_value(0xff9a, 0)
        game_area = tetris.game_area()
        inputs = np.asarray(game_area)

        # Convert blank areas into 0 and block into 1
        inputs = (inputs != blank_tile).astype(np.int16)

        try:
            shortest, tallest, holes, blank_row, min_height_diff, \
                max_height_diff, piece, next_piece, x_pos, y_pos, mean_height, \
                median_height, peaks, height_diffs, num_pieces = get_board_info(inputs, pyboy)

            inputs = np.concatenate(
                ([shortest, tallest, min_height_diff, max_height_diff, mean_height, median_height, blank_row,
                  piece, next_piece, x_pos, y_pos], holes, peaks, height_diffs, num_pieces)
            )
        except:
            output = np.array([0, 0, 0, 0, 1])
        else:

            output = population.models[child].forward(inputs)

        # if output.argmax() == 0:
        #     pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
        #     output_count['down'] += 1
        #     pyboy.tick()
        #     pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN)
        if output.argmax() == 0:
            pyboy.send_input(WindowEvent.PRESS_ARROW_LEFT)
            output_count['left'] += 1
            pyboy.tick()
            pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT)
        elif output.argmax() == 1:
            pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            output_count['a'] += 1
            pyboy.tick()
            pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
        elif output.argmax() == 2:
            pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
            output_count['right'] += 1
            pyboy.tick()
            pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
        else:
            pyboy.tick()
            # Game over
            child += 1
            population.fitnesses = np.append(population.fitnesses, tetris.fitness + tetris.lines*10)

            print("-" * 20)
            print("Iteration %s - child %s" % (epoch, child))
            print("Fitness: %s" % population.fitnesses)
            print(output_count)
            output_count = {
                'left': 0,
                'right': 0,
                'down': 0,
                'a': 0
            }
            tetris.reset_game()

    print("Iteration %s max fitness %s " % (epoch, np.max(population.fitnesses)))
    if np.max(population.fitnesses) > max_fitness:
        max_fitness = np.max(population.fitnesses)
    epoch += 1

print("Max fitness:", max_fitness)
pyboy.stop()
