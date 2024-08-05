from collections import OrderedDict

import torch

import flwr as fl

from model.training_utils import train, test

class CifarClient(fl.client.NumPyClient):

    def __init__(self, device, nn, trainloader, testloader, num_examples) -> None:
        super(CifarClient, self).__init__()
        self.device = device
        self.nn = nn
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.nn.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.nn.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.nn.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.device, self.nn, self.trainloader, epochs=1)
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.device, self.nn, self.testloader)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}