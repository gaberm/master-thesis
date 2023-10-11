from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, get_scheduler, AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch
import evaluate
from tqdm.auto import tqdm
import lang2vec.lang2vec as l2v
import numpy as np
import wandb

## testing the model

# testing parameter
best_epoch = 0
metric = evaluate.load("f1")

model_name = "bert-base-multilingual-cased"
checkpoint_path = f"bert_checkpoint_epoch_{best_epoch}.pt"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.load_state_dict(torch.load(checkpoint_path))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

target_languages = ['ar', 'bg', 'de', 'el', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
en_features = l2v.get_features("en", "syntax_knn")["en"]
clt_score = {"val_score" : [], "language_similarity" : []}

model.eval()
for lang in target_languages:
    eval_dataloader = dataloaders[lang]
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    val_score = metric.compute(average="micro")["f1"]

    lang_features = l2v.get_features(lang, "syntax_knn")[lang]
    language_similarity = np.corrcoef(en_features, lang_features)[0,1]
    clt_score["val_score"].append(val_score)
    clt_score["language_similarity"].append(language_similarity)


corr_coef = np.corrcoef(clt_score["language_similarity"], clt_score["val_score"])
print("# language similiarity")
for lang, score in zip(target_languages, clt_score["language_similarity"]):
    print(f"{lang}: {score}")
print("\n# validation score")
for lang, score in zip(target_languages, clt_score["val_score"]):
    print(f"{lang}: {score}")
print("\n# correlation coefficient")
print(corr_coef)