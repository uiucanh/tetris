import torch
import argparse

from core.gen_algo import Network
from tetris import eval_network


def main(model_path, runs):

    state_dict = torch.load(model_path)
    model = Network()
    model.load_state_dict(state_dict)

    score = eval_network(
        epoch=0,
        child_index=0,
        child_model=model,
        run_per_child=runs,
        max_score=999999,
        show=True,
        action_limit=999999,
    )

    print("Average:", score)


def cli_args():
    parser = argparse.ArgumentParser("Tetris model player")
    parser.add_argument(
        'model_path',
        help="Path to model",
        type=str,
    )
    parser.add_argument(
        '-r',
        '--runs',
        help="Tetris games played",
        type=int,
        default=1,
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = cli_args()
    main(
        model_path=args.model_path,
        runs=args.runs,
    )
