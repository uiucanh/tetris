import sys
import io
import pickle
import numpy as np

from core.utils import check_needed_dirs, check_needed_turn, do_action, \
    do_turn, do_sideway, drop_down
from core.gen_algo import get_score
from pyboy import PyBoy

n_plays = 10

quiet = "--quiet" in sys.argv
pyboy = PyBoy('tetris_1.1.gb', game_wrapper=True,
              window_type="headless" if quiet else "SDL2")
pyboy.set_emulation_speed(0)

tetris = pyboy.game_wrapper()
tetris.start_game()

# Set block animation to fall instantly
pyboy.set_memory_value(0xff9a, 2)
with open('neat_models/999999', 'rb') as f:
    model = pickle.load(f)
scores = []
lines = []
n = 0

while n < n_plays:
    # Beginning of action
    best_child_score = np.NINF
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
        score = get_score(tetris.game_area(), model,
                          tetris, s_lines, neat=True)
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
                          tetris, s_lines, neat=True)
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
                          tetris, s_lines, neat=True)
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
