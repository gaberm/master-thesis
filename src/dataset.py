import os
import platform
import pandas as pd
from datasets import load_dataset, load_from_disk, Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from .model import load_tokenizer


def prepare_paws_x(dataset):
    prepared_paws_x = dataset.rename_column("label", "labels")
    prepared_paws_x = prepared_paws_x.remove_columns(["id"])

    return prepared_paws_x


def prepare_copa(dataset):
    question_lst = []
    choice_lst = []
    label_lst = []
    for row in dataset:
        # we ignore the mirrored rows from the balanced_copa dataset
        # the mirroed rows are not contained in the original copa dataset
        if row["mirrored"] is False:
            if row["question"] == "cause":
                question = f"{row['premise']} What was the cause?"
            else:
                question = f"{row['premise']} What was the effect?"
            # add the question twice to match the number of choices
            question_lst += [question] * 2
            choice_lst += [row["choice1"]] + [row["choice2"]]
            if row["label"] == 0:
                label_lst += [1] + [0]
            else:
                label_lst += [0] + [1]

    prepared_copa = Dataset.from_dict(
        {"sentence1": question_lst,
         "sentence2": choice_lst,
         "labels": label_lst}
    )
    return prepared_copa


def prepare_siqa(dataset):
    question_lst = []
    choice_lst = []
    label_lst = []
    for row in dataset:
        # we ignore the mirrored rows from the balanced_copa dataset
        # the mirroed rows are not contained in the original copa dataset
        question = f"{row['context']} {row['question']}"
        # add the question three times to match the number of choices
        question_lst += [question] * 3
        choice_lst += [row["answerA"]] + [row["answerB"]] + [row["answerC"]]
        if row["label"] == 1:
            label_lst += [1] + [0] + [0]
        elif row["label"] == 2:
            label_lst += [0] + [1] + [0]
        else:
            label_lst += [0] + [0] + [1]

    prepared_siqa = Dataset.from_dict(
        {"sentence1": question_lst,
         "sentence2": choice_lst,
         "labels": label_lst}
    )
    return prepared_siqa


def prepare_xcopa(dataset, lang):
    question_lst = []
    choice_lst = []
    label_lst = []
    translated_question = pd.read_csv("rsc/translation.csv")
    for row in dataset:
        question_in_lang = translated_question[translated_question["lang"] == lang][row["question"]].tolist()[0]
        if row["question"] == "cause":
            question = f"{row['premise']} {question_in_lang}"
        else:
            question = f"{row['premise']} {question_in_lang}"
        question_lst += [question] * 2
        choice_lst += dataset["choice1"] + dataset["choice2"]
        if row["label"] == 0:
            label_lst += [1] + [0]
        else:
            label_lst += [0] + [1]

    prepared_xcopa = Dataset.from_dict(
        {"sentence1" : question_lst, 
         "sentence2" : choice_lst,
         "labels" : label_lst}
    )
    
    return prepared_xcopa


def prepare_xnli(dataset):
    old_col_names = ["premise", "hypothesis", "label"]
    new_col_names = ["sentence1", "sentence2", "labels"]
    for old_col, new_col in zip(old_col_names, new_col_names):
        dataset = dataset.rename_column(old_col, new_col)

    return dataset


def tokenize_ds(dataset, tokenizer):
    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True, padding=True)

    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.remove_columns(["sentence1", "sentence2"])
    dataset.set_format("torch")

    return dataset


def tokenize_function(example, tokenizer):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True, padding=True)


def download_ds(dataset, lang, split, data_dir):
    not_downloaded = True
    if dataset == "pkavumba/balanced-copa":
        try:
            os.makedirs(f"{data_dir}/datasets/balanced-copa/{lang}/{split}")
        except OSError:
            not_downloaded = False
    else:
        try:
            os.makedirs(f"{data_dir}/datasets/{dataset}/{lang}/{split}")
        except OSError:
            not_downloaded = False

    if not_downloaded:
        if dataset == "pkavumba/balanced-copa":
            ds = load_dataset(dataset, split=split)
            ds.save_to_disk(f"{data_dir}/datasets/balanced-copa/en/{split}")
        elif dataset == "social_i_qa":
            ds = load_dataset(dataset, split=split)
            ds.save_to_disk(f"{data_dir}/datasets/{dataset}/en/{split}")
        else:
            ds = load_dataset(dataset, lang, split=split)
            ds.save_to_disk(f"{data_dir}/datasets/{dataset}/{lang}/{split}")


def get_data_loader(config, split):
    data_dir = config.data_dir[platform.system().lower()]
    tokenizer = load_tokenizer(config)
    data_collator = DataCollatorWithPadding(tokenizer)

    if split == "train" or split == "val":
        if config.dataset.name == "copa":
            download_ds("pkavumba/balanced-copa", "en", split, data_dir)
            download_ds("social_i_qa", "en", split, data_dir)
            datasets = [
                load_from_disk(f"{data_dir}/datasets/balanced-copa/en/{split}"),
                load_from_disk(f"{data_dir}/datasets/social_i_qa/en/{split}")
            ]
            datasets = [
                prepare_copa(datasets[0]),
                prepare_siqa(datasets[1])
            ]
            datasets = [
                tokenize_ds(datasets[0], tokenizer),
                tokenize_ds(datasets[1], tokenizer)
            ]
            data_loaders = [DataLoader(
                ds,
                batch_size=config.params.batch_size,
                shuffle=False,
                collate_fn=data_collator,
            ) 
            for ds in datasets]
            return data_loaders
        
        if config.dataset.name == "xnli":
            download_ds("xnli", "en", split, data_dir)
            xnli = load_from_disk(f"{data_dir}/datasets/xnli/en/{split}")
            xnli = prepare_xnli(xnli)
            xnli = tokenize_ds(xnli, tokenizer)
            data_loader = DataLoader(
                xnli,
                batch_size=config.params.batch_size,
                shuffle=True,
                collate_fn=data_collator,
            )
            return data_loader
        
        if config.dataset.name == "paws_x":
            download_ds("paws-x", "en", split, data_dir)
            paws_x = load_from_disk(f"{data_dir}/datasets/paws_x/en/{split}")
            paws_x = prepare_paws_x(paws_x)
            paws_x = tokenize_ds(paws_x, tokenizer)
            data_loader = DataLoader(
                paws_x,
                batch_size=config.params.batch_size,
                shuffle=True,
                collate_fn=data_collator,
            )
            return data_loader

    if split == "test":
        test_loaders = []
        for lang in config.dataset.lang:
            if config.dataset.name == "xcopa":
                download_ds("xcopa", lang, split, data_dir)
                xcopa = load_from_disk(f"{data_dir}/datasets/xcopa/{lang}/{split}")
                xcopa = prepare_xcopa(xcopa, lang)
                xcopa = tokenize_ds(xcopa, tokenizer)
                data_loader = DataLoader(
                    xcopa,
                    batch_size=config.params.batch_size,
                    shuffle=False,
                    collate_fn=data_collator,
                )
                test_loaders.append(data_loader)

            if config.dataset.name == "xnli":
                download_ds("xnli", lang, split, data_dir)
                xnli = load_from_disk(f"{data_dir}/datasets/xnli/{lang}/{split}")
                xnli = prepare_xnli(xnli)
                xnli = tokenize_ds(xnli, tokenizer)
                data_loader = DataLoader(
                    xnli,
                    batch_size=config.params.batch_size,
                    shuffle=True,
                    collate_fn=data_collator,
                )
                test_loaders.append(data_loader)
                
            if config.dataset.name == "paws_x":
                download_ds("paws-x", lang, split, data_dir)
                paws_x = load_from_disk(f"{data_dir}/datasets/paws-x/{lang}/{split}")
                paws_x = prepare_paws_x(paws_x)
                paws_x = tokenize_ds(paws_x, tokenizer)
                data_loader = DataLoader(
                    paws_x,
                    batch_size=config.params.batch_size,
                    shuffle=True,
                    collate_fn=data_collator,
                )
                test_loaders.append(data_loader)
        
        return test_loaders