import os
from datasets import load_dataset, load_from_disk, Dataset
import pandas as pd


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


def load_ds(dataset, lang, split, data_dir):
    if dataset == "pkavumba/balanced-copa":
        ds = load_from_disk(f"{data_dir}/datasets/balanced-copa/en/{split}")
    elif dataset == "social_i_qa":
        ds = load_from_disk(f"{data_dir}/datasets/{dataset}/en/{split}")
    else:
        ds = load_from_disk(f"{data_dir}/datasets/{dataset}/{lang}/{split}")

    return ds