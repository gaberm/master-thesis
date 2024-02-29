import os
import re
import platform
import torch

def get_best_checkpoint(ckpt_dir):
    files = []
    # get all files in the checkpoint directory
    for filename in os.listdir(ckpt_dir):
        if os.path.isfile(os.path.join(ckpt_dir, filename)):
            files.append(filename)

    # get the ckpt file with the highest prediction score
    pred_scores = [float(re.findall(r"0.\d{1,3}", file)[0]) for file in files]
    best_ckpt = f"{ckpt_dir}/{files[pred_scores.index(max(pred_scores))]}"
    
    return best_ckpt


def get_device(config):
    if platform.system() == "Darwin":
        return torch.device("mps")
    if platform.system() == "Linux":
        devices = config.trainer.devices[platform.system().lower()]
        print(type(devices))
        if type(devices) == list:
            device_num = devices[0]
        else:
            device_num = devices
        return torch.device(f"cuda:{device_num}")
    else:
        raise ValueError("System not supported.")