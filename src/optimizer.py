import torch.optim as optim

def load_optimizer(model, optimizer, lr):
    match optimizer:
        case "Adam":
            return optim.Adam(model.parameters(), lr)
        case _:
            raise ValueError("Unsupported optimizer")
