from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, get_scheduler, AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch
import evaluate
from tqdm.auto import tqdm
import lang2vec.lang2vec as l2v
import numpy as np
import wandb

languages = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
datasets = {language : load_dataset("xnli", language) for language in languages}

checkpoint = "bert-base-multilingual-cased"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["premise"], example["hypothesis"], truncation=True, padding=True)

# tokenizing the dataset
dataloaders = {}
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
for language in languages:
    if language == "en":
        tokenized_dataset = datasets[language].map(tokenize_function, batched=True)
    else:
        tokenized_dataset = datasets[language]["validation"].map(tokenize_function, batched=True)

    tokenized_dataset = tokenized_dataset.remove_columns(["premise", "hypothesis"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    if language == "en":
        en_train = tokenized_dataset["train"]
        train_dataloader = DataLoader(en_train, shuffle=True, batch_size=32, collate_fn=data_collator)
        eval_dataloader = DataLoader(tokenized_dataset["validation"], batch_size=32, collate_fn=data_collator)
        dataloader = {"train" : train_dataloader, "validation" : eval_dataloader}
    else:
        dataloader = DataLoader(tokenized_dataset, batch_size=32, collate_fn=data_collator)

    dataloaders[language] = dataloader

# training settings
train_dataloader = dataloaders["en"]["train"] # finetuning the model in English
eval_dataloader = dataloaders["en"]["validation"]

optimizer = AdamW(model.parameters(), lr=5e-5)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

num_epochs = 2
num_training_steps = num_epochs * len(train_dataloader)
#lr_scheduler = get_scheduler(
#    "linear",
#    optimizer=optimizer,
#    num_warmup_steps=0,
#    num_training_steps=num_training_steps,
#)
progress_bar = tqdm(range(num_training_steps))

metric = evaluate.load("f1")
best_val_score = 0
best_epoch = 0

# wandb
wandb.login()
wandb.init(project='Sprint 2')

## training the model

model.train() # changing model to training mode
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        
        # part that pytorch lightning does for us
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        #lr_scheduler.step()
        optimizer.zero_grad()
        
        
        progress_bar.update(1)
        running_loss += loss.item()
        
        if batch_idx % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, batch_idx + 1, running_loss / 2000))
            wandb.log({'epoch': epoch+1, 'loss': running_loss/2000})
            running_loss = 0.0

    # validating the model after each epochs
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    val_score = metric.compute(average="micro")["f1"]
    print(f"F1 score epoch {epoch}: {val_score}")

    if val_score > best_val_score:
        checkpoint_path = f"bert_checkpoint_epoch_{epoch}.pt"
        model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        #torch.save(model.state_dict(), checkpoint_path)
        best_val_score = val_score
        best_epoch = epoch

print(f"The best model is from epoch: {best_epoch}.")

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