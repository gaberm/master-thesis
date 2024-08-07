import os
import pandas as pd

def main():
    all_results = []
    for file in os.listdir("results/csv"):
        if os.path.isfile(os.path.join("results/csv", file)):
            all_results.append(file)
    
    final_df = pd.DataFrame(columns=["exp_name", "task", "model", "setup", "ckpt_avg", "calib", "seed", "target_lang", "metric", "score"])
    for file in all_results:
        df = pd.read_csv(f"results/csv/{file}")
        final_df = pd.concat([final_df, df], axis=0).reset_index(drop=True)

    final_df.to_csv("results/all_results.csv", index=False)


if __name__ == "__main__":
    main()