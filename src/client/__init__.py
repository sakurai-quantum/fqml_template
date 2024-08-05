
import torch
import flwr as fl

from .cifar_client import CifarClient
from model.classical_nn import ClassicalNN
from model.training_utils import load_data

def start():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classical_nn = ClassicalNN().to(DEVICE)
    trainloader, testloader, num_examples = load_data()
    fl.client.start_client(server_address="[::]:8080", client=CifarClient(DEVICE, classical_nn, trainloader, testloader, num_examples).to_client())