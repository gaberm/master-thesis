from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
from omegaconf import DictConfig, OmegaConf
import pandas as pd

def create_sentence_pairs(dataset: DatasetDict, cfg_set_name: str, lang: str, split: str) -> DatasetDict:
    # for some datasets we need to combine columns to create sentence pairs
    # the reason for this is that the datasets have a different number of classes
    sentence_list = []
    if cfg_set_name == "balanced_copa":
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
 
        new_dataset = Dataset.from_dict({"question" : question_list, 
                                         "choice" : choice_list,
                                         "label" : label_list})
        
    elif cfg_set_name == "xcopa":
        # load translation for creating the questions
        translated_question = pd.read_csv("translation.csv")
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

        new_dataset = Dataset.from_dict({"question" : question_list, 
                                         "choice" : choice_list,
                                         "label" : label_list})
        
    elif cfg_set_name == "social_i_qa":
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

        new_dataset = Dataset.from_dict({"question" : question_list,  
                                         "answer" : answer_list,
                                         "label" : label_list})
    
    else:
        new_dataset = dataset[split]

    return new_dataset


def tokenize_and_clean_dataset(dataset: DatasetDict, set_name: str, lang: str, split: str, config: DictConfig, tokenize_function) -> DataLoader:
    # create sentence pairs
    new_dataset = create_sentence_pairs(dataset, set_name, lang, split)
    # tokenize dataset
    tokenized_set = new_dataset.map(tokenize_function, batched=True)
    # remove non-tokenized columns
    # model only expects "input_ids", "token_type_ids", "attention_mask" and "labels"
    exclude_list = list(config.dataset[set_name].columns_to_tokenize)
    tokenized_set = tokenized_set.remove_columns(exclude_list)
    # rename "label" column to "labels" if necessary
    # model except the label column to be named "labels"
    if "labels" in tokenized_set.column_names:
        tokenized_set = tokenized_set.rename_column("label", "labels")
    # set format
    tokenized_set.set_format("torch")
    print(dataset)
    return tokenized_set


def create_data_loaders(config: DictConfig, tokenizer: AutoTokenizer) -> tuple[list[DataLoader]]:

    # load datasets
    dataset_dict = {}
    for set_cfg_name in config.dataset:
        set_languages = config.dataset[set_cfg_name].languages
        dataset_name = config.dataset[set_cfg_name].name
        dataset = {lang : load_dataset(dataset_name, lang) for lang in set_languages}
        dataset_dict[set_cfg_name] = dataset

    train_set_list, val_set_list = [], []
    test_loader_dict = {}
    
    source_lang = config.params.source_lang
    data_collator = DataCollatorWithPadding(tokenizer)
    
    # tokenize and clean datasets
    for set_cfg_name, dataset in dataset_dict.items():
        # 
        def tokenize_function(example):
            return tokenizer(example[config.dataset[set_cfg_name].columns_to_tokenize[0]], 
                             example[config.dataset[set_cfg_name].columns_to_tokenize[1]], 
                             truncation=True, 
                             padding=True)
        
        for lang, dataset_in_lang in dataset.items():
            if lang == source_lang:
                # zero-shot cross lingual transfer
                # we only create train and val sets for the source language
                train_split = config.dataset[set_cfg_name].train_split
                val_split = config.dataset[set_cfg_name].val_split
                train_set = tokenize_and_clean_dataset(dataset_in_lang, set_cfg_name, lang, train_split, config, tokenize_function)
                val_set = tokenize_and_clean_dataset(dataset_in_lang, set_cfg_name, lang, val_split, config, tokenize_function)
                train_set_list.append(train_set)
                val_set_list.append(val_set)
            else:
                # we only create dataloaders for the target languages
                test_split = config.dataset[set_cfg_name].test_split
                test_loader = tokenize_and_clean_dataset(dataset_in_lang, set_cfg_name, lang, test_split, config, tokenize_function)
                loader = DataLoader(test_loader, shuffle=True, batch_size=config.params.batch_size, collate_fn=data_collator, num_workers=7)
                test_loader_dict[lang] = loader

    # merge training and validation datasets and create dataloaders
    train_val_loaders = []
    for list in [train_set_list, val_set_list]:
        merged_dataset = concatenate_datasets(list)
        loader = DataLoader(merged_dataset, batch_size=config.params.batch_size, collate_fn=data_collator, num_workers=7)
        train_val_loaders.append(loader)

    return train_val_loaders[0], train_val_loaders[1], test_loader_dict
                
#         category = config.dataset[dataset].category
#         source_language = config.params.source_lang
#         if 
#         if category == "train":
#             train_loaders.append(tokenize_for_crosslingual_transfer(config, dataset, tokenizer))
#         elif category == "test":
#             val_loaders.append(tokenize_for_crosslingual_transfer(config, dataset, tokenizer))
#         elif category == "train_test":
#             test_loaders.append(tokenize_for_crosslingual_transfer(config, dataset, tokenizer))
        
#     for train_set in train_set_list:
#         for lang, lang_dataset in train_set.items():
#             if lang == src_lang:
#                 split = config.train_datasets[set_name].source_lang_split
#             else:
#                 split = config.train_datasets[set_name].target_lang_split
#             for s in split:
#                 def tokenize_dataset(dataset: DatasetDict, dataset_name: str, split: str) -> DataLoader:
#                     tokenized_dataset = dataset[split].map(tokenize_function, batched=batched)
#                     # remove columns
#                     columns_to_remove = list(config.dataset[dataset_name].columns_to_remove)
#                     tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)
#                     # rename columns
#                     original_column_names = list(config.dataset[dataset_name].original_column_names)
#                     new_column_names = list(config.dataset[dataset_name].new_column_names)
#                     for og_name, new_name in zip(original_column_names, new_column_names):
#                         tokenized_dataset = tokenized_dataset.rename_column(og_name, new_name)
#                     # set format
#                     tokenized_dataset.set_format("torch")
#                     # create dataloader
#                     dataloader = DataLoader(tokenized_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator)

#                     return dataloader
    
#     def tokenize_dataset(dataset: list[dict[str, DatasetDict]], tokenizer: AutoTokenizer, source_language: str) -> list[DataLoader]:
#         for lang, lang_dataset in dataset.items():
#             if lang == source_language:
#                 split = config.train_datasets[set_name].source_lang_split
#             else:
#                 split = config.train_datasets[set_name].target_lang_split
# def load_datasets(config: DictConfig) -> dict[str, list[DatasetDict]]:
#     datasets_dict = {}
    
#     for ds_name in config.dataset:
#         set_languages = list(config.dataset[ds_name].languages)
#         dataset = {lang : load_dataset(ds_name, lang) for lang in set_languages}
#         datasets_dict[ds_name] = dataset
#     return datasets_dict

# def tokenize_for_crosslingual_transfer(config: DictConfig, datasets: dict[str, DatasetDict], tokenizer: AutoTokenizer) -> list[DataLoaderContainer]:
#     source_language = config.params.source_lang
#     for ds_name in datasets.keys():
#         languages_lst = config.dataset[ds_name].languages
#         batched = config.params.batched
#         batch_size = config.params.batch_size
#         data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
#         def tokenize_function(example):
#             return tokenizer(example["premise"], example["hypothesis"], truncation=config.params.truncation, padding=config.params.padding)
        
#         data_loaders = []
#         for language in languages_lst:
#             if language == source_language:
#                 split = config.dataset[ds_name].source_lang_split
#             else:
#                 split = config.dataset[ds_name].target_lang_split
#             for s in split:
#                 tokenized_dataset = datasets[ds_name][language][s].map(tokenize_function, batched=batched)
#                 def clean_dataset(dataset: DatasetDict, set_name: str) -> DatasetDict: 
#                     columns_to_remove = list(config.dataset[ds_name].columns_to_remove)
#                     tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)
#                     original_column_names = list(config.dataset[ds_name].original_column_names)
#                     new_column_names = list(config.dataset[ds_name].new_column_names)
#                     for og_name, new_name in zip(original_column_names, new_column_names):
#                         tokenized_dataset = tokenized_dataset.rename_column(og_name, new_name)
#                     return tokenized_dataset
#                 tokenized_dataset.set_format("torch")
#                 dataloader = DataLoader(tokenized_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
#                 loader = DataLoaderContainer(ds_name, language, s, dataloader)
#                 data_loaders.append(loader)
#         return data_loaders