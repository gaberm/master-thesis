import os
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk, Dataset, concatenate_datasets
import pandas as pd
import platform

# def create_sentence_pairs(dataset, dataset_name, lang, split):
#     # for some datasets we need to combine columns to create sentence pairs
#     # the reason for this is that the datasets have a different number of classes
#     sentence_list = []
#     match dataset_name:
#         case "balanced_copa":
#             for premise, question in zip(dataset[split]["premise"], dataset[split]["question"]):
#                 if question == "cause":
#                     sentence = premise + " " + "What was the cause?"
#                 elif question == "effect":
#                     sentence = premise + " " + "What was the effect?"
#                 sentence_list.append(sentence)
#             question_list = sentence_list + sentence_list 
#             choice1_list = [choice1 for choice1 in dataset[split]["choice1"]]
#             choice2_list = [choice2 for choice2 in dataset[split]["choice2"]]
#             choice_list = choice1_list + choice2_list
#             label_list = dataset[split]["label"] + dataset[split]["label"]
#             new_dataset = Dataset.from_dict({"sentence1" : question_list, 
#                                             "sentence2" : choice_list,
#                                             "labels" : label_list})
        
#         case "xcopa":
#             # load translation for creating the questions
#             translated_question = pd.read_csv("rsc/translation.csv")
#             for premise, question in zip(dataset[split]["premise"], dataset[split]["question"]):
#                 if question == "cause":
#                     sentence = premise + translated_question[translated_question["lang"] == lang]["cause"].tolist()[0]
#                 elif question == "effect":
#                     sentence = premise + translated_question[translated_question["lang"] == lang]["effect"].tolist()[0]
#                 sentence_list.append(sentence)
#             question_list = sentence_list + sentence_list 
#             choice1_list = [choice1 for choice1 in dataset[split]["choice1"]]
#             choice2_list = [choice2 for choice2 in dataset[split]["choice2"]]
#             choice_list = choice1_list + choice2_list
#             label_list = dataset[split]["label"] + dataset[split]["label"]
#             new_dataset = Dataset.from_dict({"sentence1" : question_list, 
#                                             "sentence2" : choice_list,
#                                             "labels" : label_list})
        
#         case "social_i_qa":
#             for context, question in zip(dataset[split]["context"], dataset[split]["question"]):
#                 sentence = context + " " + question
#                 sentence_list.append(sentence)
#             question_list = sentence_list + sentence_list + sentence_list
#             answer_list = []
#             for answer in ["answerA", "answerB", "answerC"]:
#                 lst = [answer for answer in dataset[split][answer]]
#                 answer_list += lst
#             label_list = []
#             for old_label in [1, 2, 3]:
#                 lst = [1 if label == old_label else 0 for label in dataset[split]["label"]]
#                 label_list += lst
#             new_dataset = Dataset.from_dict({"sentence1" : question_list,  
#                                             "sentence2" : answer_list,
#                                             "labels" : label_list})
        
#         case "xnli":
#             new_dataset = dataset[split]
#             old_columns = ["premise", "hypothesis", "label"]
#             new_columns = ["sentence1", "sentence2", "labels"]
#             for old_column, new_column in zip(old_columns, new_columns):
#                 new_dataset = new_dataset.rename_column(old_column, new_column)

#         case "paws_x":
#             new_dataset = dataset[split]
#             new_dataset= new_dataset.rename_column("label", "labels")
#             new_dataset = new_dataset.remove_columns(["id"])
        
#         case _:
#             ValueError(f"Dataset {dataset_name} not supported.")

#     return new_dataset


# def tokenize_and_clean_dataset(dataset, dataset_name, lang, split, tokenize_function):
#     # create sentence pairs
#     new_dataset = create_sentence_pairs(dataset, dataset_name, lang, split)
    
#     # tokenize dataset
#     tokenized_set = new_dataset.map(tokenize_function, batched=True)
    
#     # remove non-tokenized columns and all other remaining columns, 
#     # because model only expects "input_ids", "token_type_ids", "attention_mask" and "labels"
#     tokenized_set = tokenized_set.remove_columns(["sentence1", "sentence2"])
    
#     # set format
#     tokenized_set.set_format("torch")
    
#     return tokenized_set


# def find_local_dataset(dataset_name, data_dir, lang=None):
#     # check if dataset is saved locally
#     # some datasets only have one language, so we need to check if the language is None
#     if lang is None:
#         return os.path.exists(data_dir + f"/datasets/{dataset_name}")
#     else:
#         return os.path.exists(data_dir + f"/datasets/{dataset_name}/{lang}")
    

# def load_from_hf_or_disk(config):
#     data_dir = config.data_dir[platform.system().lower()]

#     # create directory for datasets if it doesn't exist
#     if not os.path.exists(data_dir + "/datasets"):
#         os.makedirs(data_dir + "/datasets")
    
#     # load datasets from HuggingFace Hub or locally
#     datasets_dict = {}
#     for dataset_name in config.dataset:
#         dataset = {}
#         set_languages = config.dataset[dataset_name].languages
#         hf_name = config.dataset[dataset_name].hf_name
#         if len(set_languages) == 1:
#             saved_locally = find_local_dataset(dataset_name, data_dir)
#             if saved_locally:
#                 set = load_from_disk(data_dir + f"/datasets/{dataset_name}")
#             else:
#                 set = load_dataset(hf_name)
#                 set.save_to_disk(data_dir + f"/datasets/{dataset_name}")
#             datasets_dict[dataset_name] = {set_languages[0] : set}
#         else:
#             for lang in set_languages:
#                 saved_locally = find_local_dataset(dataset_name, data_dir, lang)
#                 if saved_locally:
#                     set = load_from_disk(data_dir + f"/datasets/{dataset_name}/{lang}")
#                 else:
#                     set = load_dataset(hf_name, lang)
#                     set.save_to_disk(data_dir + f"/datasets/{dataset_name}/{lang}")
#                 dataset[lang] = set
#             datasets_dict[dataset_name] = dataset
    
#     return datasets_dict
    

# def create_train_loader(config, tokenizer):
#     datasets_dict = load_from_hf_or_disk(config)
#     source_lang = config.params.source_lang
#     data_collator = DataCollatorWithPadding(tokenizer)
    
#     def tokenize_function(example):
#         return tokenizer(example["sentence1"], example["sentence2"], truncation=True, padding=True)

#     # tokenize and clean the loaded datasets
#     train_set_lst, val_set_lst = [], []
#     for dataset_name, dataset in datasets_dict.items():
#         for lang, dataset_in_lang in dataset.items():
#             # when state is train, we create train and val loaders for the source language
#             if lang == source_lang:
#                 train_split = config.dataset[dataset_name].train_split
#                 val_split = config.dataset[dataset_name].val_split
#                 train_set = tokenize_and_clean_dataset(dataset_in_lang, dataset_name, lang, train_split, tokenize_function)
#                 val_set = tokenize_and_clean_dataset(dataset_in_lang, dataset_name, lang, val_split, tokenize_function)
#                 train_set_lst.append(train_set)
#                 val_set_lst.append(val_set)

#     # because we can have multiple datasets for training, we must concatenate the val and train sets
#     # we want to have one concatenated dataloader for training and one for validation
#     loader_lst = []
#     for set_list in [train_set_lst, val_set_lst]:
#         merged_dataset = concatenate_datasets(set_list)
#         loader = DataLoader(merged_dataset, 
#                             batch_size=config.params.batch_size, 
#                             collate_fn=data_collator, 
#                             num_workers=7)
#         loader_lst.append(loader)

#     return loader_lst[0], loader_lst[1]


# def create_test_loader(config, tokenizer):
#     dataset_name, dataset = load_from_hf_or_disk(config).popitem()
#     source_lang = config.params.source_lang
#     data_collator = DataCollatorWithPadding(tokenizer)

#     def tokenize_function(example):
#         return tokenizer(example["sentence1"], example["sentence2"], truncation=True, padding=True)
    
#     lang_loader_dict = {}
#     for lang, dataset_in_lang in dataset.items():
#         # we only test on languages that are not the source language and have an lang adapter available
#         if lang != source_lang:
#             test_split = config.dataset[dataset_name].test_split
#             test_loader = tokenize_and_clean_dataset(dataset_in_lang, dataset_name, lang, test_split, tokenize_function)
#             loader = DataLoader(test_loader, 
#                                 shuffle=True, 
#                                 batch_size=config.params.batch_size, 
#                                 collate_fn=data_collator)
#             lang_loader_dict[lang] = loader
    
#     return lang_loader_dict





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
            choice_lst += row["choice1"] + row["choice2"]
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
        choice_lst += row["choice1"] + row["choice2"] + row["choice3"]
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
    old_column_names = ["premise", "hypothesis", "label"]
    new_column_names = ["sentence1", "sentence2", "labels"]
    for old_column, new_column in zip(old_column_names, new_column_names):
        prepared_xnli = dataset.rename_column(old_column, new_column)

    return prepared_xnli


def tokenize_ds(dataset, tokenizer):
    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True, padding=True)

    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.remove_columns(["sentence1", "sentence2"])
    dataset.set_format("torch")

    return dataset


def tokenize_function(example, tokenizer):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True, padding=True)


def load_ds(dataset, lang, split, data_dir):
    from_disk = True
    if not os.path.exists(f"{data_dir}/datasets/{dataset}/{lang}"):
        os.mkdir(f"{data_dir}/datasets/{dataset}/{lang}")
        from_disk = False
    
    # we load and save the dataset to disk because its faster than
    # loading the dataset from the cache
    if from_disk:
        if dataset == "balanced_copa" or dataset == "social_i_qa":
            ds = load_from_disk(f"{data_dir}/datasets/{dataset}/en/{split}")
        else:
            ds = load_from_disk(f"{data_dir}/datasets/{dataset}/{lang}/{split}")
    else:
        if dataset == "balanced_copa" or dataset == "social_i_qa":
            ds = load_dataset(dataset, split)
            ds.save_to_disk(f"{data_dir}/datasets/{dataset}/en/{split}")
        else:
            ds = load_dataset(dataset, lang, split)
            ds.save_to_disk(f"{data_dir}/datasets/{dataset}/{lang}/{split}")
