import io
import pickle
import numpy as np
import torch

from datetime import datetime
from core.gen_algo import get_score, Population
from core.utils import check_needed_turn, do_action, drop_down, \
    do_sideway, do_turn, check_needed_dirs
from pyboy import PyBoy
from multiprocessing import Pool


epochs = 30
population = None
run_per_child = 1
max_fitness = 0
blank_tile = 47
pop_size = 30
max_score = 999999
n_workers = 10


def eval_genome(epoch, child_index, child_model):
    pyboy = PyBoy('tetris_1.1.gb', game_wrapper=True, window_type="headless")
    pyboy.set_emulation_speed(0)
    tetris = pyboy.game_wrapper()
    tetris.start_game()

    # Set block animation to fall instantly
    pyboy.set_memory_value(0xff9a, 2)

    run = 0
    scores = []
    levels = []
    lines = []

    while run < run_per_child:
        # Beginning of action
        best_action_score = np.NINF
        best_action = {'Turn': 0, 'Left': 0, 'Right': 0}
        begin_state = io.BytesIO()
        begin_state.seek(0)
        pyboy.save_state(begin_state)
        s_lines = tetris.lines

        # Determine how many possible rotations we need to check for the block
        block_tile = pyboy.get_memory_value(0xc203)
        turns_needed = check_needed_turn(block_tile)
        lefts_needed, rights_needed = check_needed_dirs(block_tile)

        # Do middle
        for move_dir in do_action('Middle', pyboy, n_dir=1,
                                  n_turn=turns_needed):
            score = get_score(tetris.game_area(), child_model,
                              tetris, s_lines)
            if score is not None and score >= best_action_score:
                best_action_score = score
                best_action = {'Turn': move_dir['Turn'],
                               'Left': move_dir['Left'],
                               'Right': move_dir['Right']}
            begin_state.seek(0)
            pyboy.load_state(begin_state)

        # Do left
        for move_dir in do_action('Left', pyboy, n_dir=lefts_needed,
                                  n_turn=turns_needed):
            score = get_score(tetris.game_area(), child_model,
                              tetris, s_lines)
            if score is not None and score >= best_action_score:
                best_action_score = score
                best_action = {'Turn': move_dir['Turn'],
                               'Left': move_dir['Left'],
                               'Right': move_dir['Right']}
            begin_state.seek(0)
            pyboy.load_state(begin_state)

        # Do right
        for move_dir in do_action('Right', pyboy, n_dir=rights_needed,
                                  n_turn=turns_needed):
            score = get_score(tetris.game_area(), child_model,
                              tetris, s_lines)
            if score is not None and score >= best_action_score:
                best_action_score = score
                best_action = {'Turn': move_dir['Turn'],
                               'Left': move_dir['Left'],
                               'Right': move_dir['Right']}
            begin_state.seek(0)
            pyboy.load_state(begin_state)

        # Do best action
        for _ in range(best_action['Turn']):
            do_turn(pyboy)
        for _ in range(best_action['Left']):
            do_sideway(pyboy, 'Left')
        for _ in range(best_action['Right']):
            do_sideway(pyboy, 'Right')
        drop_down(pyboy)
        pyboy.tick()

        # Game over:
        if tetris.game_over() or tetris.score == max_score:
            scores.append(tetris.score)
            levels.append(tetris.level)
            lines.append(tetris.lines)
            if run == run_per_child - 1:
                pyboy.stop()
            else:
                tetris.reset_game()
            run += 1

    child_fitness = np.average(scores)
    print("-" * 20)
    print("Iteration %s - child %s" % (epoch, child_index))
    print("Score: %s, Level: %s, Lines %s" %
          (scores, levels, lines))
    print("Fitness: %s" % child_fitness)
    print("Output weight:")
    print(child_model.output.weight.data)

    return child_fitness


if __name__ == '__main__':
    e = 0
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

        p = Pool(n_workers)
        result = [0] * pop_size
        for i in range(pop_size):
            result[i] = p.apply_async(
                eval_genome, (e, i, population.models[i]))

        for i in range(pop_size):
            population.fitnesses[i] = result[i].get()

        print("-" * 20)
        print("Iteration %s fitnesses %s" %
              (e, np.round(population.fitnesses, 2)))
        print(
            "Iteration %s max fitness %s " % (e, np.max(population.fitnesses)))
        print(
            "Iteration %s mean fitness %s " % (e, np.mean(
                population.fitnesses)))
        print("Time took", datetime.now() - start_time)
        print("Best child output weights:")
        print(population.models[np.argmax(
            population.fitnesses)].output.weight.T)
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
