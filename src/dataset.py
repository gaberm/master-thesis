import os
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk, Dataset, concatenate_datasets
import pandas as pd
import platform

def create_sentence_pairs(dataset, dataset_name, lang, split):
    # for some datasets we need to combine columns to create sentence pairs
    # the reason for this is that the datasets have a different number of classes
    sentence_list = []
    match dataset_name:
        case "balanced_copa":
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
        
        case "xcopa":
            # load translation for creating the questions
            translated_question = pd.read_csv("res/translation.csv")
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
        
        case "social_i_qa":
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
        
        case "xnli":
            new_dataset = dataset[split]
            old_columns = ["premise", "hypothesis", "label"]
            new_columns = ["sentence1", "sentence2", "labels"]
            for old_column, new_column in zip(old_columns, new_columns):
                new_dataset = new_dataset.rename_column(old_column, new_column)

        case "paws_x":
            new_dataset = dataset[split]
            new_dataset= new_dataset.rename_column("label", "labels")
            new_dataset = new_dataset.remove_columns(["id"])
        
        case _:
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


def find_local_dataset(dataset_name, data_dir, lang=None):
    # check if dataset is saved locally
    # some datasets only have one language, so we need to check if the language is None
    if lang is None:
        return os.path.exists(data_dir + f"/datasets/{dataset_name}")
    else:
        return os.path.exists(data_dir + f"/datasets/{dataset_name}/{lang}")
    

def create_data_loaders(config, tokenizer):
    # set data directory depending on the OS
    data_dir = config.data_dir_mac if platform.system() == "Darwin" else config.data_dir_linux

    if not os.path.exists(data_dir + "/datasets"):
        os.makedirs(data_dir + "/datasets")

    # load datasets from HuggingFace Hub or locally
    datasets_dict = {}
    for dataset_name in config.dataset:
        dataset = {}
        set_languages = config.dataset[dataset_name].languages
        hf_name = config.dataset[dataset_name].hf_name
        if len(set_languages) == 1:
            saved_locally = find_local_dataset(dataset_name, data_dir)
            if saved_locally:
                set = load_from_disk(data_dir + f"/datasets/{dataset_name}")
            else:
                set = load_dataset(hf_name)
                set.save_to_disk(data_dir + f"/datasets/{dataset_name}")
            datasets_dict[dataset_name] = {set_languages[0] : set}
        else:
            for lang in set_languages:
                saved_locally = find_local_dataset(dataset_name, data_dir, lang)
                if saved_locally:
                    set = load_from_disk(data_dir + f"/datasets/{dataset_name}/{lang}")
                else:
                    set = load_dataset(hf_name, lang)
                    set.save_to_disk(data_dir + f"/datasets/{dataset_name}/{lang}")
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
            state = config.dataset[dataset_name].state
            # when state is train, we create train and val loaders for the source language
            if lang == source_lang and state == "train":
                train_split = config.dataset[dataset_name].train_split
                val_split = config.dataset[dataset_name].val_split
                train_set = tokenize_and_clean_dataset(dataset_in_lang, dataset_name, lang, train_split, tokenize_function)
                val_set = tokenize_and_clean_dataset(dataset_in_lang, dataset_name, lang, val_split, tokenize_function)
                train_set_list.append(train_set)
                val_set_list.append(val_set)
            # when state is test, we create test loaders for target languages
            if lang != source_lang and state == "test":
                test_split = config.dataset[dataset_name].test_split
                test_loader = tokenize_and_clean_dataset(dataset_in_lang, dataset_name, lang, test_split, tokenize_function)
                loader = DataLoader(test_loader, shuffle=True, batch_size=config.params.batch_size, collate_fn=data_collator)
                test_loader_dict[lang] = loader
                return (test_loader_dict)

    # because we can have multiple datasets for training, we must concatenate the val and train sets
    # we want to have one concatenated dataloader for training and one for validation
    train_val_loaders = []
    for list in [train_set_list, val_set_list]:
        merged_dataset = concatenate_datasets(list)
        loader = DataLoader(merged_dataset, batch_size=config.params.batch_size, collate_fn=data_collator, num_workers=7)
        train_val_loaders.append(loader)
    return (train_val_loaders[0], train_val_loaders[1])