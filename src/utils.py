import os
import re

def get_best_checkpoint(ckpt_dir, with_file_extension=True):
    files = []
    # get all files in the checkpoint directory
    for filename in os.listdir(ckpt_dir):
        if os.path.isfile(os.path.join(ckpt_dir, filename)):
            files.append(filename)

    # get the ckpt file with the highest prediction score
    pred_scores = [float(re.findall(r"0.\d{1,3}", file)[0]) for file in files]
    best_ckpt = f"{ckpt_dir}/{files[pred_scores.index(max(pred_scores))]}"
    
    if with_file_extension:
        return best_ckpt
    else:
        return best_ckpt.split(".")[0]