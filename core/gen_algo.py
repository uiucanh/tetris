import numpy as np
import torch
import torch.nn as nn
from core.utils import get_board_info
from torch.autograd import Variable

input_size = 4
hidden_size = 4
output_size = 1


class Network(nn.Module):
    def __init__(self, hidden_w=None, output_w=None):
        super(Network, self).__init__()
        if not hidden_w:
            self.hidden = nn.Linear(input_size, hidden_size).cuda()
            self.hidden.weight.requires_grad_(False)
            torch.nn.init.uniform_(self.hidden.weight, a=-1, b=1)
        else:
            self.hidden = hidden_w

        if not output_w:
            self.output = nn.Linear(hidden_size, output_size).cuda()
            self.output.weight.requires_grad_(False)
            torch.nn.init.uniform_(self.output.weight, a=-1, b=1)
        else:
            self.output = output_w

        # Define sigmoid activation and softmax output
        self.sigmoid = nn.Sigmoid().cuda()

    def forward(self, x):
        with torch.no_grad():
            x = torch.from_numpy(x).float().to('cuda')
            x = self.hidden(x)
            x = self.sigmoid(x)
            x = self.output(x)

        return x


class Population:
    def __init__(self, size=10, old_population=None):
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
        self.fitnesses = np.array([])

    def crossover(self):
        print("Crossver")
        self.old_fitnesses = np.square(self.old_fitnesses)
        sum_fitnesses = np.sum(self.old_fitnesses)
        probs = [self.old_fitnesses[i] / sum_fitnesses for i in
                 range(self.size)]
        # Add the top 50% performing childs
        sort_indices = np.argsort(probs)[::-1]
        for i in range(self.size):
            if i < self.size / 2:
                model_c = self.old_models[sort_indices[i]]
            else:
                # a = np.random.choice(self.old_models, size=1, p=probs,
                #                      replace=False)[0]
                a, b = np.random.choice(self.size, size=2, p=probs,
                                        replace=False)
                sum_parent = self.old_fitnesses[a] + self.old_fitnesses[b]
                prob_neuron_from_a = \
                    self.old_fitnesses[a] / sum_parent

                model_a, model_b = self.old_models[a], self.old_models[b]
                model_c = Network()

                for j in range(hidden_size):
                    if np.random.random() > prob_neuron_from_a:
                        model_c.hidden.weight.data[j] = \
                            model_b.hidden.weight.data[j]
                    else:
                        model_c.hidden.weight.data[j] = \
                            model_a.hidden.weight.data[j]

                for j in range(output_size):
                    if np.random.random() > prob_neuron_from_a:
                        model_c.output.weight.data[j] = \
                            model_b.output.weight.data[j]
                    else:
                        model_c.output.weight.data[j] = \
                            model_a.output.weight.data[j]

            self.models.append(model_c)
            # c = Network()
            #
            # for j in range(len(a.hidden.weight.data)):
            #     if j % 2 == 0:
            #         c.hidden.weight.data[i] = a.hidden.weight.data[i]
            #         continue
            #     c.hidden.weight.data[i] = b.hidden.weight.data[i]
            # assert a != c
            # self.models.append(c)

            # a, b = np.random.choice(self.old_models, size=2, p=probs,
            #                         replace=True)
            #
            # new_hidden = torch.cat(
            #     (a.hidden.weight.data[0:int(hidden_size / 2)],
            #      b.hidden.weight.data[int(hidden_size / 2):])
            # )
            # new_output = torch.cat(
            #     (a.output.weight.data[0:int(output_size / 2)],
            #      b.output.weight.data[int(output_size / 2):])
            # )
            #
            # with torch.no_grad():
            #     a.hidden.weight.data = new_hidden
            #     a.output.weight.data = new_output

            # self.models.append(a)

    def mutate(self):
        print("Mutating")
        for model in self.models:
            for i in range(len(model.hidden.weight.data)):
                if np.random.random() < 0.05:
                    # mutation = Variable(
                    #     model.hidden.weight.data[i].new(
                    #         model.hidden.weight.data[i].size())).uniform_()
                    # model.hidden.weight.data[i] = mutation
                    with torch.no_grad():
                        noise = torch.randn(
                            model.hidden.weight.data[i].size()).mul_(
                            0.5).cuda()
                        model.hidden.weight.data[i].add_(noise)

            for i in range(len(model.output.weight.data)):
                if np.random.random() < 0.05:
                    # mutation = Variable(model.output.weight.data[i].new(
                    #     model.output.weight.data[i].size())).uniform_()
                    # model.output.weight.data[i] = mutation
                    with torch.no_grad():
                        noise = torch.randn(
                            model.output.weight.data[i].size()).mul_(
                            0.5).cuda()
                        model.output.weight.data[i].add_(noise)


def get_score(area, model, pyboy):
    area = np.asarray(area)

    # Convert blank areas into 0 and block into 1
    area = (area != 47).astype(np.int16)

    try:
        inputs = get_board_info(area, pyboy)
    except Exception as e:
        print(e)
        return None

    return model.forward(inputs)
