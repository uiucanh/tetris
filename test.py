#
# License: See LICENSE.md file
# GitHub: https://github.com/Baekalfen/PyBoy
#

import os
import sys
from core.utils import *

# Makes us able to import PyBoy from the directory below
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + "/..")

from pyboy import PyBoy, WindowEvent # isort:skip

quiet = "--quiet" in sys.argv
pyboy = PyBoy('tetris_1.gb', window_type="headless" if quiet else "SDL2", game_wrapper=True)
pyboy.set_emulation_speed(1)

tetris = pyboy.game_wrapper()
tetris.start_game()


blank_tile = 47
first_brick = False

while not pyboy.tick():  # Enough frames for the test. Otherwise do: `while not pyboy.tick():`
    s_lines = pyboy.get_memory_value(0xff9e)

    if pyboy.paused:
        area = np.asarray(tetris.game_area())
        area = (area != 47).astype(np.int16)
        agg_height, e_lines, n_holes, bumpiness = get_board_info(area,
                                                                 pyboy, s_lines)
        print({
            'agg_height': agg_height,
            'n_holes': n_holes,
            'e_lines': e_lines,
            'bumpiness': bumpiness})
        #     # 'min_cons_height_diff': min_cons_height_diff,
        #     # 'max_cons_height_diff': max_cons_height_diff,
        #     # 'max_height_diff': max_height_diff,
        #     # 't_lines_cleared': t_lines_cleared, 'e_lines': e_lines,
        #     # # 'num_pieces': num_pieces, 'mean_height': mean_height,
        #     # # 'median_height': median_height,
        #     # 'n_col_with_holes': n_col_with_holes,
        #     # 'sum_weighted': sum_weighted,
        #     # 'max_col_holes': max_col_holes,
        #     # # 'sum_cons_height_diff': sum_cons_height_diff,
        #     # 'num_pits': num_pits
        # })

    # print("lines_cleared", pyboy.get_memory_value(0xff9e))
    # print("lines", tetris.lines)
    # pyboy.tick()
    # print(pyboy.get_memory_value(0xff9a))
    # print(pyboy.get_memory_value(0xff99))
    # print("piece:", pyboy.get_memory_value(0xc203))
    # print("X: %s, Y: %s" % (pyboy.get_memory_value(0xc202) - 39, (pyboy.get_memory_value(0xc201) - 24) / 8))
    # print("X Of rightmost lowest piece block", pyboy.get_memory_value(0xff92) - 39)
    # print("Y Of rightmost lowest piece block", (pyboy.get_memory_value(0xff93) - 24) / 8)
    # pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
    # pyboy.tick()
    # pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN)

    # game_area is accessed as [<row>, <column>].
    # 'game_area[-1,:]' is asking for all (:) the columns in the last row (-1)
    # if frame % 5 == 0:
    #     shortest, tallest, n_holes, blank_row, peaks, height_diffs = get_score(inputs)
    #     print(inputs)
    #     print("shortest", shortest)
    #     print("tallest", tallest)
    #     print("holes", n_holes)
    #     print("blank", blank_row)
    #     print("peaks", peaks)
    #     print("height_diffs", height_diffs)

# import pickle
# with open("test.pkl", "wb") as f:
#     pickle.dump(inputs, f)


# print("Final game board mask:")
# print(tetris)

# We shouldn't have made any progress with the moves we made
# try:
# assert tetris.score == 0
# assert tetris.level == 0
# assert tetris.lines == 0
# assert tetris.fitness == 0 # A built-in fitness score for AI developmen
# except :

# Assert there is something on the bottom of the game area
# assert any(filter(lambda x: x != blank_tile, game_area[-1, :]))
# tetris.reset_game()
# # After reseting, we should have a clean game area
# assert all(filter(lambda x: x != blank_tile, game_area[-1, :]))

pyboy.stop()