import torch as T

class Hyper:
    total_epochs = 1
    learning_rate = 1e-6
    batch_size = 2

    [staticmethod]   
    def display():
        print("The Hyperparameters")
        print("-------------------")
        print(f"NUmber of epochs = {Hyper.total_epochs}")
        print(f"learning rate = {Hyper.learning_rate}")
        print(f"batch_size = {Hyper.batch_size}")

class Constants:
    device = T.device("cuda" if T.cuda.is_available() else "cpu")