import os
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, concatenate_datasets
from omegaconf import DictConfig, OmegaConf
import pandas as pd

def create_sentence_pairs(dataset, dataset_name, lang, split):
    # for some datasets we need to combine columns to create sentence pairs
    # the reason for this is that the datasets have a different number of classes
    sentence_list = []
    if dataset_name == "balanced_copa":
        for premise, question in zip(dataset[split]["premise"], dataset[split]["question"]):
            if question == "cause":
                sentence = premise + " " + "What is the cause?"
            elif question == "effect":
                sentence = premise + " " + "What is the effect?"
            sentence_list.append(sentence)
        
        question_list = sentence_list + sentence_list 
        choice1_list = [choice1 for choice1 in dataset[split]["choice1"]]
        choice2_list = [choice2 for choice2 in dataset[split]["choice2"]]
        choice_list = choice1_list + choice2_list
        label_list = dataset[split]["label"] + dataset[split]["label"]
 
        new_dataset = Dataset.from_dict({"sentence1" : question_list, 
                                         "sentence2" : choice_list,
                                         "labels" : label_list})
        
    elif dataset_name == "xcopa":
        # load translation for creating the questions
        translated_question = pd.read_csv("data/translation.csv")
        for premise, question in zip(dataset[split]["premise"], dataset[split]["question"]):
            if question == "cause":
                sentence = premise + translated_question[translated_question["lang"] == lang]["cause"].tolist()[0]
            elif question == "effect":
                sentence = premise + translated_question[translated_question["lang"] == lang]["effect"].tolist()[0]
            sentence_list.append(sentence)

        question_list = sentence_list + sentence_list 
        choice1_list = [choice1 for choice1 in dataset[split]["choice1"]]
        choice2_list = [choice2 for choice2 in dataset[split]["choice2"]]
        choice_list = choice1_list + choice2_list
        label_list = dataset[split]["label"] + dataset[split]["label"]

        new_dataset = Dataset.from_dict({"sentence1" : question_list, 
                                         "sentence2" : choice_list,
                                         "labels" : label_list})
        
    elif dataset_name == "social_i_qa":
        for context, question in zip(dataset[split]["context"], dataset[split]["question"]):
            sentence = context + " " + question
            sentence_list.append(sentence)

        question_list = sentence_list + sentence_list + sentence_list
        answer_list = []
        for answer in ["answerA", "answerB", "answerC"]:
            lst = [answer for answer in dataset[split][answer]]
            answer_list += lst
        label_list = []
        for old_label in [1, 2, 3]:
            lst = [1 if label == old_label else 0 for label in dataset[split]["label"]]
            label_list += lst

        new_dataset = Dataset.from_dict({"sentence1" : question_list,  
                                         "sentence2" : answer_list,
                                         "labels" : label_list})
        
    elif dataset_name == "xnli":
        new_dataset = dataset[split]
        old_columns = ["premise", "hypothesis", "label"]
        new_columns = ["sentence1", "sentence2", "labels"]
        for old_column, new_column in zip(old_columns, new_columns):
            new_dataset = new_dataset.rename_column(old_column, new_column)

    elif dataset_name == "paws_x":
        new_dataset = dataset[split]
        new_dataset= new_dataset.rename_column("label", "labels")
        new_dataset = new_dataset.remove_columns(["id"])
        
    else:
        ValueError(f"Dataset {dataset_name} not supported.")

    return new_dataset


def tokenize_and_clean_dataset(dataset, dataset_name, lang, split, tokenize_function):
    # create sentence pairs
    new_dataset = create_sentence_pairs(dataset, dataset_name, lang, split)
    
    # tokenize dataset
    tokenized_set = new_dataset.map(tokenize_function, batched=True)
    
    # remove non-tokenized columns and all other remaining columns, 
    # because model only expects "input_ids", "token_type_ids", "attention_mask" and "labels"
    tokenized_set = tokenized_set.remove_columns(["sentence1", "sentence2"])
    
    # set format
    tokenized_set.set_format("torch")
    
    return tokenized_set


def find_local_dataset(dataset_name, lang=None):
    # check if dataset is saved locally
    # some datasets only have one language, so we need to check if the language is None
    if lang is None:
        return os.path.exists(f"data/datasets/{dataset_name}")
    else:
        return os.path.exists(f"data/datasets/{dataset_name}/{lang}")
    

def create_data_loaders(config, tokenizer):
    # load datasets from HuggingFace Hub or locally
    datasets_dict = {}
    for dataset_name in config.dataset:
        dataset = {}
        set_languages = config.dataset[dataset_name].languages
        hf_name = config.dataset[dataset_name].hf_name
        if len(set_languages) == 1:
            saved_locally = find_local_dataset(dataset_name)
            if saved_locally:
                set = load_from_disk(f"data/datasets/{dataset_name}")
            else:
                set = load_dataset(hf_name)
                set.save_to_disk(f"data/datasets/{dataset_name}")
            datasets_dict[dataset_name] = {set_languages[0] : set}
        else:
            for lang in set_languages:
                saved_locally = find_local_dataset(dataset_name, lang)
                if saved_locally:
                    set = load_from_disk(f"data/datasets/{dataset_name}/{lang}")
                else:
                    set = load_dataset(hf_name, lang)
                    set.save_to_disk(f"data/datasets/{dataset_name}/{lang}")
                dataset[lang] = set
            datasets_dict[dataset_name] = dataset

    train_set_list, val_set_list = [], []
    test_loader_dict = {}
    
    source_lang = config.params.source_lang
    data_collator = DataCollatorWithPadding(tokenizer)
    
    # tokenize and clean the loaded datasets
    for dataset_name, dataset in datasets_dict.items():
        # 
        def tokenize_function(example):
            return tokenizer(example["sentence1"], example["sentence2"], truncation=True, padding=True)
        
        for lang, dataset_in_lang in dataset.items():
            # check if dataset is used for training
            train = config.dataset[dataset_name].train
            if lang == source_lang and train:
                # zero-shot cross lingual transfer
                # we only create train and val sets for the source language
                train_split = config.dataset[dataset_name].train_split
                val_split = config.dataset[dataset_name].val_split
                train_set = tokenize_and_clean_dataset(dataset_in_lang, dataset_name, lang, train_split, tokenize_function)
                val_set = tokenize_and_clean_dataset(dataset_in_lang, dataset_name, lang, val_split, tokenize_function)
                train_set_list.append(train_set)
                val_set_list.append(val_set)
            else:
                # we only create dataloaders for the target languages
                test_split = config.dataset[dataset_name].test_split
                test_loader = tokenize_and_clean_dataset(dataset_in_lang, dataset_name, lang, test_split, tokenize_function)
                loader = DataLoader(test_loader, shuffle=True, batch_size=config.params.batch_size, collate_fn=data_collator)
                test_loader_dict[lang] = loader

    # merge training and validation datasets and create dataloaders
    train_val_loaders = []
    for list in [train_set_list, val_set_list]:
        merged_dataset = concatenate_datasets(list)
        loader = DataLoader(merged_dataset, batch_size=config.params.batch_size, collate_fn=data_collator, num_workers=7)
        train_val_loaders.append(loader)

    return train_val_loaders[0], train_val_loaders[1], test_loader_dict