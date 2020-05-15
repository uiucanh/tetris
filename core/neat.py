import numpy as np

from core.utils import get_board_info


def get_score(area, model, pyboy):
    area = np.asarray(area)

    # Convert blank areas into 0 and block into 1
    area = (area != 47).astype(np.int16)

    try:
        inputs = (get_board_info(area, pyboy, concat=False))
    except Exception as e:
        print(e)
        return None

    return model.activate(inputs)[0]
