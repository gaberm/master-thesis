import os
import re
import shutil
import torch
import omegaconf
from tqdm import tqdm
import numpy as np
import pandas as pd


def find_best_ckpt(ckpt_dir):
    all_ckpts = []
    for file in os.listdir(ckpt_dir):
        if os.path.isfile(os.path.join(ckpt_dir, file)):
            all_ckpts.append(file)

    # find the checkpoint with the highest validation score (in the source language)
    val_scores = [float(re.findall(r"0\.\d{1,3}", ckpt)[0]) for ckpt in all_ckpts]
    best_ckpt = f"{ckpt_dir}/{all_ckpts[val_scores.index(max(val_scores))]}"
    
    return best_ckpt


def get_device(config):
    devices = config.trainer.devices
    if isinstance(devices, omegaconf.listconfig.ListConfig):
        device_num = devices[0]
    else:
        device_num = devices
    
    return torch.device(f"cuda:{device_num}")
    

def save_test_results(model, config, seed):
    try:
        os.makedirs(f"results/{config.trainer.exp_name}")
    except OSError:
        pass
    np.save(f"results/{config.trainer.exp_name}/seed_{seed}", model.result_lst)
    

def create_result_csv(exp_name):
    mslt = True if "mslt" in exp_name else False # mslt = multi-source language training
    schmidt = True if "schmidt" in exp_name else False
    files = []
    result_dir = f"results/{exp_name}"
    for filename in os.listdir(result_dir):
        if os.path.isfile(os.path.join(result_dir, filename)):
            files.append(filename)

    # we merge the results list of all seeds into one dataframe
    if mslt:
        col_lst = ["exp_name", "dataset", "model", "setup", "source_lang", "target_lang", "seed", "metric", "score"]
    else:
        col_lst = ["exp_name", "dataset", "model", "setup", "ca_strategy", "calib", "seed", "source_lang", "target_lang", "metric", "score"]
    final_df = pd.DataFrame(columns=col_lst)
    for file in files:
        df = pd.DataFrame(np.load(f"{result_dir}/{file}", allow_pickle=True), columns=col_lst)
        final_df = pd.concat([final_df, df], axis=0).reset_index(drop=True)
    final_df['score'] = final_df['score'].astype(float)

    if mslt:
        try:
            os.makedirs("results/mslt/csv")
        except OSError:
            pass
        final_df.to_csv(f"results/mslt/csv/{exp_name}.csv", index=False)
    elif schmidt:
        try:
            os.makedirs("results/schmidt/csv")
        except OSError:
            pass
        final_df.to_csv(f"results/schmidt/csv/{exp_name}.csv", index=False)
    else:
        try:
            os.makedirs("results/sslt/csv")
        except OSError:
            pass
        final_df.to_csv(f"results/sslt/csv/{exp_name}.csv", index=False)

    # remove all result lists by deleting the result directory
    # try except block to avoid errors if the system runs on multiple threads
    try:
        shutil.rmtree(result_dir)
    except OSError:
        pass


def compute_ckpt_average(ckpt_dir, device, ca_strategy, ckpts_to_load):
    n = ckpts_to_load
    all_ckpts = []
    for file in os.listdir(ckpt_dir):
        if os.path.isfile(os.path.join(ckpt_dir, file)):
            all_ckpts.append(file)

    # best: take the 5 checkpoints with the highest validation score (in the source language)
    if ca_strategy == "best":
        val_scores = [float(re.findall(r"0\.\d{1,3}", ckpt)[0]) for ckpt in all_ckpts]
        idx = np.argsort(val_scores)[-n:]
        final_ckpts = [all_ckpts[i] for i in idx]
    
    # last: take the last checkpoint of each epoch
    if ca_strategy == "last":
        val_scores = [float(re.findall(r"0\.\d{1,3}", ckpt)[0]) for ckpt in all_ckpts]
        final_ckpts = []
        # ckpt files are sorted by step, so we can just iterate through them
        for i in range(n):
            epoch_ckpts = [ckpt for ckpt in all_ckpts if f"epoch={i}" in ckpt]
            step_scores = [int(re.findall(r"step=\d+", ckpt)[0].replace("step=", "")) for ckpt in epoch_ckpts]
            idx = np.argsort(step_scores)[-1]
            final_ckpts.append(epoch_ckpts[idx])
    
    k = len(final_ckpts)
    average_state_dict = None
    for ckpt in tqdm(final_ckpts, total=k, desc="Loading checkpoints"):
        ckpt = torch.load(f"{ckpt_dir}/{ckpt}", map_location=device)["state_dict"]
        if average_state_dict is None:
            average_state_dict = ckpt
            for key, value in average_state_dict.items():
                if value.is_floating_point():
                    average_state_dict[key] = value / k
        else:
            for key, value in ckpt.items():
                if value.is_floating_point():
                    average_state_dict[key] += value / k
        del ckpt

    if "xcopa" in ckpt_dir or "xsc" in ckpt_dir:
        enc_state_dict = {
            k.replace("encoder.", "", 1): v 
            for k, v in average_state_dict.items() 
            if "encoder." in k
        }
        cls_state_dict = {
            k.replace("classifier.", ""): v 
            for k, v in average_state_dict.items() 
            if "classifier." in k
        }
        return enc_state_dict, cls_state_dict
    else:
        model_state_dict = {
            k.replace("model.", ""): v 
            for k, v in average_state_dict.items()
        }
        return model_state_dict