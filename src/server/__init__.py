
import flwr as fl

from server.server_strategy import SaveModelStrategy


def start():
    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=100), strategy=SaveModelStrategy())