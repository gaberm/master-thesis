import os
import re
import shutil
import torch
import omegaconf
from tqdm import tqdm
import numpy as np
import pandas as pd


def find_best_ckpt(ckpt_dir):
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
    devices = config.trainer.devices
    if isinstance(devices, omegaconf.listconfig.ListConfig):
        device_num = devices[0]
    else:
        device_num = devices
    
    return torch.device(f"cuda:{device_num}")
    

def save_test_results(model, config, seed):
    # try to create a tempory directory to save the results
    # the directory will be deleted after we have created the csv
    try:
        os.mkdir(f"res/{config.trainer.exp_name}")
    except FileExistsError:
        pass
    np.save(f"res/{config.trainer.exp_name}/seed_{seed}", model.result_lst)
    

def create_result_csv(exp_name):
    files = []
    result_dir = f"res/{exp_name}"
    for filename in os.listdir(result_dir):
        if os.path.isfile(os.path.join(result_dir, filename)):
            files.append(filename)

    # we merge the results list of all seeds into one dataframe
    final_df = pd.DataFrame(columns=["exp_name", "seed", "target_lang", "metric", "score"])
    for file in files:
        df = pd.DataFrame(
            np.load(f"{result_dir}/{file}", allow_pickle=True),
            columns=["exp_name", "seed", "target_lang", "metric", "score"]
        )
        final_df = pd.concat([final_df, df], axis=0).reset_index(drop=True)
    final_df['score'] = final_df['score'].astype(float)

    final_df.to_csv(f"res/{exp_name}.csv", sep=";", decimal=",", index=False)

    # remove all result lists by deleting the result directory
    # try except block to avoid errors if the system runs on multiple threads
    try:
        shutil.rmtree(result_dir)
    except OSError:
        pass


def compute_val_score(val_score_1, val_score_2, num_samples_1, num_samples_2):
    return (val_score_1 * num_samples_1 + val_score_2 * num_samples_2) / (num_samples_1 + num_samples_2)


def compute_ckpt_average(ckpt_dir, device):
    ckpt_files = []
    for filename in os.listdir(ckpt_dir):
        if os.path.isfile(os.path.join(ckpt_dir, filename)):
            ckpt_files.append(filename)
    
    k = len(ckpt_files)
    average_state_dict = None
    for ckpt_file in tqdm(ckpt_files, total=k, desc="Loading checkpoints"):
        ckpt = torch.load(f"{ckpt_dir}/{ckpt_file}", map_location=device)["state_dict"]
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

    if "copa" in ckpt_dir:
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