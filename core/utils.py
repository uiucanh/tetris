import numpy as np

from pyboy import WindowEvent

# Action map
action_map = {
    'Left': [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT],
    'Right': [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT],
    'Down': [WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN],
    'A': [WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A]
}

# Define some variables
blank_tile = 47
start_y = 24


def get_current_block_text(block_tile):
    if 0 <= block_tile <= 3:
        return 'L'
    elif 4 <= block_tile <= 7:
        return 'J'
    elif 8 <= block_tile <= 11:
        return 'I'
    elif 12 <= block_tile <= 15:
        return 'O'
    elif 16 <= block_tile <= 19:
        return 'Z'
    elif 20 <= block_tile <= 23:
        return 'S'
    elif 24 <= block_tile <= 27:
        return 'T'


def get_board_info(area, pyboy, concat=True):
    # Num of rows - first occurrence of 1
    peaks = np.array([])
    for col in range(area.shape[1]):
        if 1 in area[:, col]:
            p = area.shape[0] - np.argmax(area[:, col], axis=0)
            peaks = np.append(peaks, p)
        else:
            peaks = np.append(peaks, 0)

    # Shortest and tallest col
    agg_height = np.sum(peaks)

    # Number of empty holes
    holes = get_holes(peaks, area)
    n_holes = np.sum(holes)

    # Abs height differences between consecutive cols
    bumpiness = get_bumpiness(peaks)

    if not concat:
        return agg_height, n_holes, bumpiness
    return \
        np.array([agg_height, n_holes, bumpiness])


def get_board_info_for_neat(area, concat=True):
    # Num of rows - first occurrence of 1
    peaks = np.array([])
    for col in range(area.shape[1]):
        if 1 in area[:, col]:
            p = area.shape[0] - np.argmax(area[:, col], axis=0)
            peaks = np.append(peaks, p)
        else:
            peaks = np.append(peaks, 0)

    # Shortest and tallest col
    agg_height = np.sum(peaks)
    shortest = np.min(peaks)
    tallest = np.max(peaks)
    max_height_diff = tallest - shortest

    # Number of empty holes
    holes = get_holes(peaks, area)
    max_col_holes = np.max(holes)
    n_holes = np.sum(holes)
    n_col_with_holes = np.count_nonzero(holes)

    # Abs height differences between consecutive cols
    bumpiness = get_bumpiness(peaks)

    # Mean and median heights
    mean_height = np.mean(peaks)
    median_height = np.median(peaks)

    num_pieces = np.count_nonzero(area)
    # Number of cols with zero blocks
    num_pits = np.count_nonzero(np.count_nonzero(area, axis=0) == 0)

    # t_lines_cleared = pyboy.get_memory_value(0xff9e)
    # e_lines = np.count_nonzero(np.all(area == np.ones(10), axis=1))

    # Total sum of weighted blocks, the higher the block the higher the score
    sum_weighted = np.sum((area.T * np.arange(area.shape[0], 0, -1)).T)
    # sum_weighted = np.interp(sum_weighted, (0, 1710), (0, 10))

    if not concat:
        return agg_height, n_holes, bumpiness, shortest, tallest, \
               max_height_diff, max_col_holes, n_col_with_holes, num_pieces, \
               num_pits, mean_height, median_height, sum_weighted
    return \
        np.array([agg_height, n_holes, bumpiness, shortest, tallest, \
                  max_height_diff, max_col_holes, n_col_with_holes, num_pieces, \
                  num_pits, mean_height, median_height, sum_weighted])


def get_bumpiness(peaks):
    s = 0
    for i in range(9):
        s += np.abs(peaks[i] - peaks[i + 1])
    return s


def get_num_pieces_on_board(area, blank_row):
    c = []
    for col in range(area.shape[1]):
        c.append(np.count_nonzero(area[blank_row:, col]))

    return np.array(c)


def get_holes(peaks, area):
    # Count from peaks to bottom
    holes = []
    for col in range(area.shape[1]):
        start = -peaks[col]
        # If there's no holes i.e. no blocks on that column
        if start == 0:
            holes.append(0)
        else:
            holes.append(np.count_nonzero(area[int(start):, col] == 0))
    return holes


def get_min_max_col_height(board_height, blank_row, col_peaks):
    tallest = board_height - blank_row - 1
    shortest = np.min(col_peaks)

    return shortest, tallest


def count_max_consecutive_0(last_row):
    mask = np.concatenate(([False], last_row == 0, [False]))
    if ~mask.any():
        return 0
    else:
        c = np.flatnonzero(mask[1:] < mask[:-1]) - \
            np.flatnonzero(mask[1:] > mask[:-1])
        return c.max()


def check_needed_turn(block_tile):
    # Get the text representation of the block
    block = get_current_block_text(block_tile)
    if block == 'I' or block == 'S' or block == 'Z':
        return 2
    if block == 'O':
        return 1
    return 4


def check_needed_dirs(block_tile):
    # Return left, right maximum try needed
    block = get_current_block_text(block_tile)
    if block == 'S' or block == 'Z':
        return 3, 5
    if block == 'O':
        return 4, 4
    return 4, 5


def do_turn(pyboy):
    pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
    pyboy.tick()
    pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
    pyboy.tick()


def do_sideway(pyboy, action):
    pyboy.send_input(action_map[action][0])
    pyboy.tick()
    pyboy.send_input(action_map[action][1])
    pyboy.tick()


def do_down(pyboy):
    pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
    pyboy.tick()
    pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN)


def drop_down(pyboy):
    # We continue moving down until can't anymore. This will cause
    # a new piece to spawn and have y_value of start_y.
    # started_moving to prevent the loop not running at start.
    started_moving = False
    while pyboy.get_memory_value(0xc201) != start_y or not started_moving:
        started_moving = True
        do_down(pyboy)


def do_action(action, pyboy, n_dir, n_turn):
    for dir_count in range(1, n_dir + 1):
        for turn in range(1, n_turn + 1):
            # Turn
            for t in range(turn):
                do_turn(pyboy)

            # Move in direction
            if action != 'Middle':
                for move in range(dir_count):
                    do_sideway(pyboy, action)

            drop_down(pyboy)

            yield {'Turn': turn,
                   'Left': dir_count if action == 'Left' else 0,
                   'Right': dir_count if action == 'Right' else 0}
