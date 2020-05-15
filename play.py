import sys
import io

from datetime import datetime
from core.utils import *
from core.gen_algo import *
from pyboy import PyBoy, WindowEvent

max_fitness = 0
population = None
epoch = 0
pop_size = 50

quiet = "--quiet" in sys.argv
pyboy = PyBoy('tetris.gb', game_wrapper=True,
              window_type="headless" if quiet else "SDL2")
pyboy.set_emulation_speed(0)

tetris = pyboy.game_wrapper()
tetris.start_game()

# Set block animation to fall instantly
pyboy.set_memory_value(0xff9a, 1)
state_dict = torch.load('models/14_18_15_330119.0')
model = Network()
model.load_state_dict(state_dict)
block_pool = []

n_plays = 10
play = 0
scores = []

while play < n_plays:
    # block_pool = [0, 4, 8, 12, 16, 20, 24] if len(block_pool) == 0 else \
    #     block_pool
    # next_piece = np.random.choice(block_pool)
    # block_pool.remove(next_piece)
    # # Set the next piece to random
    # pyboy.set_memory_value(0xc213, next_piece)

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
    for i in range(best_action['Turn']):
        do_turn(pyboy)
    for i in range(best_action['Left']):
        do_sideway(pyboy, 'Left')
    for i in range(best_action['Right']):
        do_sideway(pyboy, 'Right')
    drop_down(pyboy)
    pyboy.tick()

    # Game over:
    if pyboy.get_memory_value(0xffe1) == 13:
        scores.append(tetris.score)
        print("Run %s, Score: %s, Level: %s, Lines %s" %
              (play, tetris.score, tetris.level,
               pyboy.get_memory_value(0xff9e)))
        play += 1
        tetris.reset_game()

print("Scores:", scores)
print("Average:", np.average(scores))
pyboy.stop()
