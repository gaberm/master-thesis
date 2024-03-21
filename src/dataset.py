import os
import platform
import random
import pandas as pd
from datasets import load_dataset, load_from_disk, Dataset
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler, Sampler
from transformers import DataCollatorWithPadding
from .model import load_tokenizer
from lightning.pytorch.utilities import CombinedLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

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
    
    ds_lst = [question_lst, choice_lst, label_lst]
    question_lst, choice_lst, label_lst = shuffle_lst(ds_lst, 2)

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
        if row["label"] == "1":
            label_lst += [1] + [0] + [0]
        elif row["label"] == "2":
            label_lst += [0] + [1] + [0]
        else:
            label_lst += [0] + [0] + [1]

    ds_lst = [question_lst, choice_lst, label_lst]
    question_lst, choice_lst, label_lst = shuffle_lst(ds_lst, 3)

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
        choice_lst += [row["choice1"]] + [row["choice2"]]
        if row["label"] == 0:
            label_lst += [1] + [0]
        else:
            label_lst += [0] + [1]

    ds_lst = [question_lst, choice_lst, label_lst]
    question_lst, choice_lst, label_lst = shuffle_lst(ds_lst, 2)

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
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True, padding=True, max_length=512)

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
    if config.dataset.name == "copa" or config.dataset.name == "xcopa": 
        data_collator = DataCollatorWithPadding(tokenizer)
    else:
        data_collator = DataCollatorWithPadding(tokenizer)

    if split == "train" or split == "validation":
        if config.dataset.name == "copa":
            if split == "train":
                download_ds("pkavumba/balanced-copa", "en", "train", data_dir)
                download_ds("social_i_qa", "en", "train", data_dir)
                datasets = [
                    load_from_disk(f"{data_dir}/datasets/balanced-copa/en/train"),
                    load_from_disk(f"{data_dir}/datasets/social_i_qa/en/train")
                ]
            if split == "validation":
                download_ds("pkavumba/balanced-copa", "en", "test", data_dir)
                download_ds("social_i_qa", "en", "validation", data_dir)
                datasets = [
                    load_from_disk(f"{data_dir}/datasets/balanced-copa/en/test"),
                    load_from_disk(f"{data_dir}/datasets/social_i_qa/en/validation")
                ]
            datasets = [
                prepare_copa(datasets[0]), 
                prepare_siqa(datasets[1])
            ]
            datasets = [tokenize_ds(ds, tokenizer) for ds in datasets]
            combined_loader = CombinedLoader(
                {"copa": DataLoader(
                    datasets[0],
                    batch_size=config.params.batch_size*2,
                    sampler=SequentialSampler(datasets[0]),
                    collate_fn=data_collator,
                ),"siqa": DataLoader(
                    datasets[1],
                    batch_size=config.params.batch_size*3,
                    sampler=SequentialSampler(datasets[1]),
                    collate_fn=data_collator,
                )},
                "max_size"
            )
            _ = iter(combined_loader) 
            return combined_loader
        
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
        for lang in config.dataset.test_lang:
            if config.dataset.name == "xcopa":
                download_ds("xcopa", lang, split, data_dir)
                xcopa = load_from_disk(f"{data_dir}/datasets/xcopa/{lang}/{split}")
                xcopa = prepare_xcopa(xcopa, lang)
                xcopa = tokenize_ds(xcopa, tokenizer)
                data_loader = DataLoader(
                    xcopa,
                    batch_size=config.params.batch_size,
                    sampler=CopaSampler(xcopa, config.params.batch_size),
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
    

class CopaSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        self.data_size = len(self.data_source)

    def __iter__(self):
        self.indices = [
            list(range(i, min(i + self.batch_size, self.data_size)))
            for i in range(0, self.data_size, self.batch_size) 
        ]
        random.shuffle(self.indices)
        self.indices = [idx for sublist in self.indices for idx in sublist]
        return iter(self.indices)
    
    def __len__(self):
        return self.data_size

def shuffle_lst(ds_lst: list[str | int], num_labels):
    batch_lst = []
    for lst in ds_lst:
        lst = [
            lst[i:i+num_labels] for i in range(0, len(lst), num_labels) 
            if (i + num_labels) < len(lst)
        ]
        batch_lst.append(lst)

    random.shuffle([zip(*batch_lst)])

    flattened_lst = []
    for lst in batch_lst:
        flattened_lst.append([sentence for batch in lst for sentence in batch])

    return flattened_lst