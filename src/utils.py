import os
import re
import platform
import torch
import omegaconf
import numpy as np
import pandas as pd


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
        if isinstance(devices, omegaconf.listconfig.ListConfig):
            device_num = devices[0]
        else:
            device_num = devices
        return torch.device(f"cuda:{device_num}")
    else:
        raise ValueError("System not supported.")
    

def create_result_csv(result_dir):
    # get all files in the checkpoint directory
    files = []
    for filename in os.listdir(result_dir):
        if os.path.isfile(os.path.join(result_dir, filename)):
            files.append(filename)

    # add all result lists for each seed together
    df_data = []
    for file in files:
        df_data += np.load(f"{result_dir}/{file}", allow_pickle=True)
    df = pd.DataFrame(df_data, columns=["seed", "target_lang", "metric", "score"])

    # save the complete result list as csv
    df.to_csv(f"{df}/test_results.csv", sep=";", decimal=",", index=False)

    # remove all result lists
    for file in files:
        os.remove(f"{result_dir}/{file}")

