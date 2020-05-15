import io

from datetime import datetime
from core.utils import *
from core.gen_algo import *
from pyboy import PyBoy
from multiprocessing import Pool, cpu_count

epochs = 10
population = None
max_fitness = 0
blank_tile = 47
pop_size = 50
max_action_count = 1000
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


def eval_genome(epoch, child_index, child_model):
    pyboy = PyBoy('tetris.gb', game_wrapper=True, window_type="headless")
    pyboy.set_emulation_speed(0)
    tetris = pyboy.game_wrapper()
    custom_start(pyboy)

    # Set block animation to fall instantly
    pyboy.set_memory_value(0xff9a, 1)
    action_count = 0

    while not pyboy.tick():
        # Beginning of action
        best_action_score = np.NINF
        best_action = {'Turn': 0, 'Left': 0, 'Right': 0}
        begin_state = io.BytesIO()
        begin_state.seek(0)
        pyboy.save_state(begin_state)

        # Determine how many possible rotations we need to check for the block
        block_tile = pyboy.get_memory_value(0xc203)
        turns_needed = check_needed_turn(block_tile)
        lefts_needed, rights_needed = check_needed_dirs(block_tile)

        # Do middle
        for move_dir in do_action('Middle', pyboy, n_dir=1,
                                  n_turn=turns_needed):
            score = get_score(tetris.game_area(), child_model,
                              pyboy)
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
                              pyboy)
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
                              pyboy)
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
        action_count += 1

        # Game over:
        if pyboy.get_memory_value(0xffe1) == 13 \
                or action_count > max_action_count:
            child_fitness = tetris.score
            print("-" * 20)
            print("Iteration %s - child %s" % (epoch, child_index))
            if action_count > max_action_count:
                print("Stop because of max action count")
            print("Score: %s, Level: %s, Lines %s" %
                  (tetris.score, tetris.level, tetris.lines))
            print("Fitness: %s" % child_fitness)
            pyboy.stop()

            return child_fitness


for e in range(epochs):
    start_time = datetime.now()
    if population is None:
        population = Population(size=pop_size)
    else:
        population = Population(size=pop_size, old_population=population)

    p = Pool(n_workers)
    result = []
    for i in range(pop_size):
        result.append(p.apply_async(eval_genome, (e, i, population.models[i])))

    for i in range(pop_size):
        population.fitnesses[i] = result[i].get()

    print("Iteration %s fitnesses %s" % (e, np.round(population.fitnesses,2)))
    print(
        "Iteration %s max fitness %s " % (e, np.max(population.fitnesses)))
    print(
        "Iteration %s mean fitness %s " % (e, np.mean(
            population.fitnesses)))
    print("Time took", datetime.now() - start_time)

    if np.max(population.fitnesses) > max_fitness:
        max_fitness = np.max(population.fitnesses)
        file_name = datetime.strftime(datetime.now(), '%d_%H_%M_') + str(
            np.round(max_fitness, 2))
        torch.save(
            population.models[np.argmax(population.fitnesses)].state_dict(),
            'models/%s' % file_name)
    e += 1
