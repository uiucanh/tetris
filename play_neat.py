import sys
import io
import pickle

from core.utils import *
from core.neat import *
from pyboy import PyBoy

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
# pyboy.set_memory_value(0xff9a, 0)
with open('neat_models/100.67', 'rb') as f:
    model = pickle.load(f)
block_pool = []

while not pyboy.tick():
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
    pyboy.save_state(begin_state)

    # Determine how many possible rotations we need to check for the block
    block_tile = pyboy.get_memory_value(0xc203)
    turns_needed = check_needed_turn(block_tile)
    lefts_needed, rights_needed = check_needed_dirs(block_tile)
    count = 0

    # Do middle
    for move_dir in do_action('Middle', pyboy, n_dir=1,
                              n_turn=turns_needed):
        count += 1
        score = get_score(tetris.game_area(), model,
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
        count += 1
        score = get_score(tetris.game_area(), model,
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
        count += 1
        score = get_score(tetris.game_area(), model,
                          pyboy)
        if score is not None and score > best_child_score:
            best_child_score = score
            best_action = {'Turn': move_dir['Turn'],
                           'Left': move_dir['Left'],
                           'Right': move_dir['Right']}
        begin_state.seek(0)
        pyboy.load_state(begin_state)

    # Do best action
    print(best_child_score)
    count = 0
    for i in range(best_action['Turn']):
        do_turn(pyboy)
    for i in range(best_action['Left']):
        do_sideway(pyboy, 'Left')
    for i in range(best_action['Right']):
        do_sideway(pyboy, 'Right')
    drop_down(pyboy)
    pyboy.tick()
    best_child_score = -np.inf

    # Game over:
    if pyboy.get_memory_value(0xffe1) == 13:
        input()
        tetris.reset_game()
