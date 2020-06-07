import torch
import numpy as np

from core.gen_algo import Network
from multiprocessing import Pool
from tetris import eval_network

state_dict = torch.load('models/best.pkl')
model = Network()
model.load_state_dict(state_dict)

n_workers = 10
n_plays = 10

result = [0] * n_plays
scores = [0] * n_plays

p = Pool(n_workers)

for i in range(n_plays):
    result[i] = p.apply_async(eval_network, (0, i, model))

for i in range(n_plays):
    scores[i] = result[i].get()

print("Scores:", scores)
print("Average:", np.average(scores))
