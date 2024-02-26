import torch.optim as optim

def load_optimizer(model, optimizer, lr):
    match optimizer:
        case "adamW":
            return optim.AdamW(model.parameters(), lr)
        case _:
            raise ValueError("Unsupported optimizer")
