import os
import random
import pandas as pd
from datasets import Dataset, load_dataset, load_from_disk
from torch.utils.data import DataLoader, SequentialSampler
from transformers import DataCollatorWithPadding
from .model import load_tokenizer
from lightning.pytorch.utilities import CombinedLoader
from sklearn.model_selection import train_test_split


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
                label_lst += [1, 0]
            else:
                label_lst += [0, 1]
    
    ds_lst = [question_lst, choice_lst, label_lst]
    question_lst, choice_lst, label_lst = shuffle_lst(ds_lst, 2)

    prepared_copa = Dataset.from_dict(
        {"sentence1": question_lst,
         "sentence2": choice_lst,
         "labels": label_lst}
    )
    return prepared_copa


def prepare_paws_x(dataset):
    prepared_paws_x = dataset.rename_column("label", "labels")
    prepared_paws_x = prepared_paws_x.remove_columns(["id"])
    return prepared_paws_x


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
            label_lst += [1, 0, 0]
        elif row["label"] == "2":
            label_lst += [0, 1, 0]
        else:
            label_lst += [0, 0, 1]

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
    translated_question = pd.read_csv("rsc/translated_questions_xcopa.csv")
    for row in dataset:
        question_in_lang = translated_question[translated_question["lang"] == lang][row["question"]].tolist()[0]
        if row["question"] == "cause":
            question = f"{row['premise']} {question_in_lang}"
        else:
            question = f"{row['premise']} {question_in_lang}"
        question_lst += [question] * 2
        choice_lst += [row["choice1"]] + [row["choice2"]]
        label_lst += [1, 0] if row["label"] == 0 else [0, 1]

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


def prepare_xstorycloze(dataset, lang):
    questions_df = pd.read_csv("rsc/translated_questions_xstorycloze.csv")
    context_lst = []
    answer_lst = []
    label_lst = []
    for row in dataset:
        question = questions_df[questions_df["lang"] == lang]["question"].tolist()[0]
        context = [row[f"InputSentence{i}"] for i in range(1, 5)]
        context_lst += ["".join(context) + f" {question}"] * 2
        answer_lst += [row["RandomFifthSentenceQuiz1"]] + [row["RandomFifthSentenceQuiz2"]]
        label_lst += [1, 0] if row["AnswerRightEnding"] == 1 else [0, 1]    

    prepared_xstorycloze = Dataset.from_dict(
        {"sentence1" : context_lst, 
         "sentence2" : answer_lst,
         "labels" : label_lst}
    )
    return prepared_xstorycloze


def tokenize_ds(dataset, tokenizer):
    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True, padding=True) 

    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.remove_columns(["sentence1", "sentence2"])
    dataset.set_format("torch")
    return dataset


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
    data_dir = config.data_dir
    tokenizer = load_tokenizer(config)
    data_collator = DataCollatorWithPadding(tokenizer)

    if split in ["train", "validation"]:
        if config.dataset.name == "paws_x":
            download_ds("paws-x", "en", split, data_dir)
            paws_x = load_from_disk(f"{data_dir}/datasets/paws-x/en/{split}")
            paws_x = prepare_paws_x(paws_x)
            paws_x = tokenize_ds(paws_x, tokenizer)
            data_loader = DataLoader(
                paws_x,
                batch_size=config.params.batch_size,
                shuffle=True,
                collate_fn=data_collator,
            )
        
        elif config.dataset.name == "xcopa":
            if split == "train":
                download_ds("social_i_qa", "en", "train", data_dir)
                download_ds("pkavumba/balanced-copa", "en", "train", data_dir)
                copa = load_from_disk(f"{data_dir}/datasets/balanced-copa/en/train")
                siqa = load_from_disk(f"{data_dir}/datasets/social_i_qa/en/train")
                siqa = prepare_siqa(siqa)
                siqa = tokenize_ds(siqa, tokenizer)
            else:
                download_ds("pkavumba/balanced-copa", "en", "test", data_dir)
                copa = load_from_disk(f"{data_dir}/datasets/balanced-copa/en/test")
            copa = prepare_copa(copa)
            copa = tokenize_ds(copa, tokenizer)
            if split == "train":
                data_loader = CombinedLoader(
                    {"copa": DataLoader(
                        copa,
                        batch_size=config.params.batch_size*2,
                        sampler=SequentialSampler(copa),
                        collate_fn=data_collator,
                    ),"siqa": DataLoader(
                        siqa,
                        batch_size=config.params.batch_size*3,
                        sampler=SequentialSampler(siqa),
                        collate_fn=data_collator,
                    )},
                    "max_size"
                )
                _ = iter(data_loader)
            else:
                data_loader = DataLoader(
                    copa,
                    batch_size=config.params.batch_size,
                    collate_fn=data_collator,
                )
        
        elif config.dataset.name == "xnli":
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

        elif config.dataset.name == "xstorycloze":
            storycoze = pd.read_csv("rsc/storycloze.csv")
            train_df, val_df = train_test_split(storycoze, test_size=0.2, random_state=42)
            if split == "train":
                download_ds("social_i_qa", "en", "train", data_dir)
                siqa = load_from_disk(f"{data_dir}/datasets/social_i_qa/en/train")
                siqa = prepare_siqa(siqa)
                siqa = tokenize_ds(siqa, tokenizer)
                storycoze = train_df
            else:
                storycoze = val_df
            storycoze = storycoze.to_dict()
            storycoze = {k: list(v.values()) for k, v in storycoze.items()}
            storycoze = Dataset.from_dict(storycoze)
            storycoze = prepare_xstorycloze(storycoze, "en")
            storycoze = tokenize_ds(storycoze, tokenizer)

            if split == "train":
                data_loader = CombinedLoader(
                    {"siqa": DataLoader(
                        siqa,
                        batch_size=config.params.batch_size*3,
                        sampler=SequentialSampler(siqa),
                        collate_fn=data_collator,
                    ),"storycloze": DataLoader(
                        storycoze,
                        batch_size=config.params.batch_size*2,
                        sampler=SequentialSampler(storycoze),
                        collate_fn=data_collator,
                    )},
                    "max_size"
                )
                _ = iter(data_loader)
            else:
                data_loader = DataLoader(
                    storycoze,
                    batch_size=config.params.batch_size*2,
                    collate_fn=data_collator,
                )
                
        else:
            raise ValueError(f"Dataset {config.dataset.name} not supported")
        
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
                    sampler=SequentialSampler(xcopa),
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

            if config.dataset.name == "xstorycloze":
                storycoze = pd.read_csv("rsc/xstorycloze.csv")
                storycoze = storycoze[storycoze["lang"] == lang]
                storycoze = storycoze.to_dict()
                storycoze = {k: list(v.values()) for k, v in storycoze.items()}
                storycoze = Dataset.from_dict(storycoze)
                storycoze = prepare_xstorycloze(storycoze, lang)
                storycoze = tokenize_ds(storycoze, tokenizer)
                data_loader = DataLoader(
                    storycoze,
                    batch_size=config.params.batch_size,
                    sampler=SequentialSampler(storycoze),
                    collate_fn=data_collator,
                )
                test_loaders.append(data_loader)
        
        return test_loaders
    

def shuffle_lst(ds_lst: list[str | int], num_labels):
    # in prepare_coqa and prepare_siqa we duplicate the same question "num_labels" times 
    # to make each question a binary classification task
    # shuffle_lst shuffles the dataset in a way that duplicates of the same question are kept in order
    # the training loop only works if the questions remain in order
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