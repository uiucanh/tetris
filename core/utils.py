import numpy as np
import io

from pyboy import PyBoy
from pyboy.utils import WindowEvent


# Action map
action_map = {
    'Left': [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT],
    'Right': [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT],
    'Down': [WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN],
    'A': [WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A]
}

start_y = 24

feature_names = [
    'agg_height', 'n_holes', 'bumpiness', 'cleared', 'num_pits', 'max_wells',
    'n_cols_with_holes', 'row_transitions', 'col_transitions'
]


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


def get_board_info(area, tetris, s_lines, next_block, neat=False):
    """
    area: a numpy matrix representation of the board
    tetris: game wrapper
    s_lines: the starting number of cleared lines
    """
    # Columns heights
    peaks = get_peaks(area)
    highest_peak = np.max(peaks)

    # Aggregated height
    agg_height = np.sum(peaks)

    holes = get_holes(peaks, area)
    # Number of empty holes
    n_holes = np.sum(holes)
    # Number of columns with at least one hole
    n_cols_with_holes = np.count_nonzero(np.array(holes) > 0)

    # Row transitions
    row_transitions = get_row_transition(area, highest_peak)

    # Columns transitions
    col_transitions = get_col_transition(area, peaks)

    # Abs height differences between consecutive cols
    bumpiness = get_bumpiness(peaks)

    # Number of cols with zero blocks
    num_pits = np.count_nonzero(np.count_nonzero(area, axis=0) == 0)

    wells = get_wells(peaks)
    # Deepest well
    max_wells = np.max(wells)

    # The number of lines gained with the move
    cleared = (tetris.lines - s_lines) * 8

    res = (
        agg_height,
        n_holes,
        bumpiness,
        cleared,
        num_pits,
        max_wells,
        n_cols_with_holes,
        row_transitions,
        col_transitions,
    )

    if neat:
        # experimental neat addition: also pass next tetromino (block) to model
        # on 3 additional inputs
        next_block_bits = {
            "L": (0, 0, 0),
            "J": (0, 0, 1),
            "I": (0, 1, 0),
            "O": (0, 1, 1),
            "Z": (1, 0, 0),
            "S": (1, 0, 1),
            "T": (1, 1, 0),
        }
        bits = next_block_bits[next_block]

        res = (
            *res,
            *bits,
        )

    return res


def get_peaks(area):
    peaks = np.array([])
    for col in range(area.shape[1]):
        if 1 in area[:, col]:
            p = area.shape[0] - np.argmax(area[:, col], axis=0)
            peaks = np.append(peaks, p)
        else:
            peaks = np.append(peaks, 0)
    return peaks


def get_row_transition(area, highest_peak):
    sum = 0
    # From highest peak to bottom
    for row in range(int(area.shape[0] - highest_peak), area.shape[0]):
        for col in range(1, area.shape[1]):
            if area[row, col] != area[row, col - 1]:
                sum += 1
    return sum


def get_col_transition(area, peaks):
    sum = 0
    for col in range(area.shape[1]):
        if peaks[col] <= 1:
            continue
        for row in range(int(area.shape[0] - peaks[col]), area.shape[0] - 1):
            if area[row, col] != area[row + 1, col]:
                sum += 1
    return sum


def get_bumpiness(peaks):
    s = 0
    for i in range(9):
        s += np.abs(peaks[i] - peaks[i + 1])
    return s


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


def get_wells(peaks):
    wells = []
    for i in range(len(peaks)):
        if i == 0:
            w = peaks[1] - peaks[0]
            w = w if w > 0 else 0
            wells.append(w)
        elif i == len(peaks) - 1:
            w = peaks[-2] - peaks[-1]
            w = w if w > 0 else 0
            wells.append(w)
        else:
            w1 = peaks[i - 1] - peaks[i]
            w2 = peaks[i + 1] - peaks[i]
            w1 = w1 if w1 > 0 else 0
            w2 = w2 if w2 > 0 else 0
            w = w1 if w1 >= w2 else w2
            wells.append(w)
    return wells


def check_needed_turn(block_tile):
    # Check how many turns we need to check for a block
    block = get_current_block_text(block_tile)
    if block == 'I' or block == 'S' or block == 'Z':
        return 2
    if block == 'O':
        return 1
    return 4


def check_needed_dirs(block_tile):
    # Return left, right moves needed
    block = get_current_block_text(block_tile)
    if block == 'S' or block == 'Z':
        return 3, 5
    if block == 'O':
        return 4, 4
    return 4, 5


def do_turn(pyboy):
    pyboy.send_input(action_map['A'][0])
    pyboy.tick()
    pyboy.send_input(action_map['A'][1])
    pyboy.tick()


def do_sideway(pyboy, action):
    pyboy.send_input(action_map[action][0])
    pyboy.tick()
    pyboy.send_input(action_map[action][1])
    pyboy.tick()


def do_down(pyboy):
    pyboy.send_input(action_map['Down'][0])
    pyboy.tick()
    pyboy.send_input(action_map['Down'][1])


def drop_down(pyboy):
    # We continue moving down until can't anymore. This will cause
    # a new piece to spawn and have y_value of start_y.
    # The bool started_moving is used to prevent the loop not running at start.
    started_moving = False
    while pyboy.memory[0xc201] != start_y or not started_moving:
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


def do_best_action(get_score, pyboy, tetris, model, neat):

    # Beginning of action
    best_child_score = -np.inf
    best_action = {'Turn': 0, 'Left': 0, 'Right': 0}
    begin_state = io.BytesIO()
    begin_state.seek(0)
    pyboy.save_state(begin_state)
    s_lines = tetris.lines

    # Determine how many possible rotations we need to check for the block
    block_tile = pyboy.memory[0xc203]
    turns_needed = check_needed_turn(block_tile)
    lefts_needed, rights_needed = check_needed_dirs(block_tile)
    next_block = tetris.next_tetromino()

    actions = {
        'Middle': 1,
        'Left': lefts_needed,
        'Right': rights_needed,
    }

    for action, n_dir in actions.items():
        for move_dir in do_action(action, pyboy, n_dir=n_dir,
                                  n_turn=turns_needed):
            score = get_score(
                tetris,
                model,
                s_lines,
                neat=neat,
                next_block=next_block,
            )
            if score is not None and score > best_child_score:
                best_child_score = score
                best_action = move_dir.copy()
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

    return best_action


def spawn_pyboy(show=False):
    window = "SDL2" if show else "null"
    pyboy = PyBoy('tetris.gb', window=window)
    pyboy.set_emulation_speed(0)
    tetris = pyboy.game_wrapper
    tetris.start_game()
    # Set block animation to fall instantly
    pyboy.memory[0xff9a] = 2
    return pyboy, tetris
