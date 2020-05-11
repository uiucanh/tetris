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
tetris.reset_game()

# Set block animation to fall instantly
pyboy.set_memory_value(0xff9a, 0)
state_dict = torch.load('models/07_21_10_388559.0')
model = Network()
model.load_state_dict(state_dict)
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
    beginning = pyboy.save_state(begin_state)
    # The starting count of lines
    # s_lines = pyboy.get_memory_value(0xff9e)

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
    print(best_child_score)

    # action_count += 1

    # Game over:
    if pyboy.get_memory_value(0xffe1) == 13:
        tetris.reset_game()
        # area = np.asarray(tetris.game_area())
        #
        # # Convert blank areas into 0 and block into 1
        # area = (area != 47).astype(np.int16)
        #
        # shortest, tallest, n_holes, \
        # max_height_diff, t_lines_cleared, \
        # e_lines, max_col_holes, \
        # n_col_with_holes, num_pits = get_board_info(area, pyboy)
        #
        # print("shortest", shortest)
        # print("tallest", tallest)
        # print("n_holes", n_holes)
        # print("max_height_diff", max_height_diff)
        # print("t_lines_cleared", t_lines_cleared)
        # print("e_lines", e_lines)
        # print("max_col_holes", max_col_holes)
        # print("n_col_with_holes", n_col_with_holes)
        # print("num_pits", num_pits)
        # input()
