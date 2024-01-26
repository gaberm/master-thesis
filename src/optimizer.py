import torch.optim as optim

def load_optimizer(model, optimizer, lr):
    if optimizer == "AdamW":
        return optim.AdamW(model.parameters(), lr)
    else:
        raise ValueError("Unsupported optimizer")
