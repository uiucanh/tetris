import numpy as np
import torch
import torch.nn as nn
from core.utils import get_board_info

input_size = 9
output_size = 1

elitism_pct = 0.2
mutation_prob = 0.2
weights_init_min = -1
weights_init_max = 1
weights_mutate_power = 0.5

device = 'cpu'


class Network(nn.Module):
    def __init__(self, output_w=None):
        super(Network, self).__init__()
        if not output_w:
            self.output = nn.Linear(
                input_size, output_size, bias=False).to(device)
            self.output.weight.requires_grad_(False)
            torch.nn.init.uniform_(self.output.weight,
                                   a=weights_init_min, b=weights_init_max)
        else:
            self.output = output_w

    def activate(self, x):
        with torch.no_grad():
            x = torch.from_numpy(x).float().to(device)
            x = self.output(x)
        return x


class Population:
    def __init__(self, size=50, old_population=None):
        self.size = size
        if old_population is None:
            self.models = [Network() for i in range(size)]
        else:
            # Copy the child
            self.old_models = old_population.models
            self.old_fitnesses = old_population.fitnesses
            self.models = []
            self.crossover()
            self.mutate()
        self.fitnesses = np.zeros(self.size)

    def crossover(self):
        print("Crossver")
        sum_fitnesses = np.sum(self.old_fitnesses)
        probs = [self.old_fitnesses[i] / sum_fitnesses for i in
                 range(self.size)]

        # Sorting descending NNs according to their fitnesses
        sort_indices = np.argsort(probs)[::-1]
        for i in range(self.size):
            if i < self.size * elitism_pct:
                # Add the top performing childs
                model_c = self.old_models[sort_indices[i]]
            else:
                a, b = np.random.choice(self.size, size=2, p=probs,
                                        replace=False)
                # sum_parent = self.old_fitnesses[a] + self.old_fitnesses[b]
                # Probability that each neuron will come from model A
                # prob_neuron_from_a = \
                #     self.old_fitnesses[a] / sum_parent
                prob_neuron_from_a = 0.5

                model_a, model_b = self.old_models[a], self.old_models[b]
                model_c = Network()

                for j in range(input_size):
                    # Neuron will come from A with probability
                    # of `prob_neuron_from_a`
                    if np.random.random() > prob_neuron_from_a:
                        model_c.output.weight.data[0][j] = \
                            model_b.output.weight.data[0][j]
                    else:
                        model_c.output.weight.data[0][j] = \
                            model_a.output.weight.data[0][j]

            self.models.append(model_c)

    def mutate(self):
        print("Mutating")
        for model in self.models:
            # Mutating weights by adding Gaussian noises
            for i in range(input_size):
                if np.random.random() < mutation_prob:
                    with torch.no_grad():
                        noise = torch.randn(1).mul_(
                            weights_mutate_power).to(device)
                        model.output.weight.data[0][i].add_(noise[0])


def get_score(tetris, model, s_lines, neat=False, next_block=None):
    area = np.asarray(tetris.game_area())
    # Convert blank areas into 0 and block into 1
    area = (area != 47).astype(np.int16)

    try:
        inputs = get_board_info(
            area,
            tetris,
            s_lines,
            next_block=next_block,
            neat=neat
        )
    except Exception as e:
        print(e)
        return None

    if neat:
        output = model.activate(inputs)[0]
    else:
        output = model.activate(np.array(inputs))
    return output
