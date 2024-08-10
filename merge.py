import os
import pandas as pd

def main():
    for finetuning in ["single_src_ft", "multi_src_ft"]:
        all_results = []
        for file in os.listdir(f"results/{finetuning}/csv"):
            if os.path.isfile(os.path.join(f"results/{finetuning}/csv", file)):
                all_results.append(file)
        
        if finetuning == "multi_src_ft":
            final_df = pd.DataFrame(columns=["exp_name", "dataset", "model", "setup", "source_lang", "target_lang", "seed", "metric", "score"])
        else:
            final_df = pd.DataFrame(columns=["exp_name", "dataset", "model", "setup", "ckpt_avg", "calib", "seed", "source_lang", "target_lang", "metric", "score"])
        for file in all_results:
            df = pd.read_csv(f"results/{finetuning}/csv/{file}")
            final_df = pd.concat([final_df, df], axis=0).reset_index(drop=True)

        final_df.to_csv(f"results/{finetuning}/full_results.csv", index=False)

if __name__ == "__main__":
    main()