import numpy as np
import torch
import torch.nn as nn
from core.utils import get_board_info

input_size = 9
# hidden_size = 1
output_size = 1

elitism_pct = 0.2
mutation_prob = 0.5
weights_init_min = -1
weights_init_max = 1
weights_mutate_power = 0.1

device = 'cpu'


class Network(nn.Module):
    def __init__(self, output_w=None):
        super(Network, self).__init__()
        # if not hidden_w:
        #     self.hidden = nn.Linear(input_size, hidden_size, bias=False).to(device)
        #     self.hidden.weight.requires_grad_(False)
        #     torch.nn.init.uniform_(self.hidden.weight,
        #                            a=weights_init_min, b=weights_init_max)
        # else:
        #     self.hidden = hidden_w

        # if not hidden_2_w:
        #     self.hidden_2 = nn.Linear(input_size, hidden_size, bias=False).to(device)
        #     self.hidden_2.weight.requires_grad_(False)
        #     torch.nn.init.uniform_(self.hidden_2.weight,
        #                            a=weights_init_min, b=weights_init_max)
        # else:
        #     self.hidden_2 = hidden_2_w

        if not output_w:
            self.output = nn.Linear(input_size, output_size, bias=False).to(device)
            self.output.weight.requires_grad_(False)
            torch.nn.init.uniform_(self.output.weight,
                                   a=weights_init_min, b=weights_init_max)
        else:
            self.output = output_w

    def activate(self, x):
        with torch.no_grad():
            x = torch.from_numpy(x).float().to(device)
            # x = self.hidden(x)
            # x = self.hidden_2(x)
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
        # Squaring all fitnesses
        # self.old_fitnesses = np.square(self.old_fitnesses)
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
                sum_parent = self.old_fitnesses[a] + self.old_fitnesses[b]
                # Probability that each neuron will come from model A
                prob_neuron_from_a = \
                    self.old_fitnesses[a] / sum_parent
                # prob_neuron_from_a = 0.5

                model_a, model_b = self.old_models[a], self.old_models[b]
                model_c = Network()

                # for j in range(hidden_size):
                #     # Neuron will come from A with probability
                #     # of `prob_neuron_from_a`
                #     if np.random.random() > prob_neuron_from_a:
                #         model_c.hidden.weight.data[j] = \
                #             model_b.hidden.weight.data[j]
                #     else:
                #         model_c.hidden.weight.data[j] = \
                #             model_a.hidden.weight.data[j]

                # for j in range(hidden_size):
                #     # Neuron will come from A with probability
                #     # of `prob_neuron_from_a`
                #     if np.random.random() > prob_neuron_from_a:
                #         model_c.hidden_2.weight.data[j] = \
                #             model_b.hidden_2.weight.data[j]
                #     else:
                #         model_c.hidden_2.weight.data[j] = \
                #             model_a.hidden_2.weight.data[j]

                for j in range(output_size):
                    # Do the same for output weights
                    if np.random.random() > prob_neuron_from_a:
                        model_c.output.weight.data[j] = \
                            model_b.output.weight.data[j]
                    else:
                        model_c.output.weight.data[j] = \
                            model_a.output.weight.data[j]

            self.models.append(model_c)

    def mutate(self):
        print("Mutating")
        for model in self.models:
            # for i in range(len(model.hidden.weight.data)):
            #     if np.random.random() < mutation_prob:
            #         with torch.no_grad():
            #             noise = torch.randn(
            #                 model.hidden.weight.data[i].size()).mul_(
            #                 weights_mutate_power).to(device)
            #             model.hidden.weight.data[i].add_(noise)

            # for i in range(len(model.hidden_2.weight.data)):
            #     if np.random.random() < mutation_prob:
            #         with torch.no_grad():
            #             noise = torch.randn(
            #                 model.hidden_2.weight.data[i].size()).mul_(
            #                 weights_mutate_power).to(device)
            #             model.hidden_2.weight.data[i].add_(noise)

            for i in range(len(model.output.weight.data)):
                if np.random.random() < mutation_prob:
                    with torch.no_grad():
                        noise = torch.randn(
                            model.output.weight.data[i].size()).mul_(
                            weights_mutate_power).to(device)
                        model.output.weight.data[i].add_(noise)


def get_score(area, model, tetris, s_lines, neat=False):
    area = np.asarray(area)
    # Convert blank areas into 0 and block into 1
    area = (area != 47).astype(np.int16)

    try:
        inputs = get_board_info(area, tetris, s_lines)
    except Exception as e:
        print(e)
        return None

    output = model.activate(inputs)
    if neat:
        output = model.activate(inputs)[0]
    else:
        output = model.activate(np.array(inputs))
    return output
