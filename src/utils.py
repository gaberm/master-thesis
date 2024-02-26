import os
import shutil
import re

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


def move_files(ckpt_dir, run_name):
    # create a new directory for the run
    run_dir = f"{ckpt_dir}/{run_name}"
    os.makedirs(run_dir)

    for filename in os.listdir(ckpt_dir):
        ckpt_file = f"{ckpt_dir}/{filename}"
        run_file = f"{run_dir}/{filename}"
        # move the file to the run directory
        shutil.move(ckpt_file, run_file)