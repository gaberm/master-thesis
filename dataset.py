from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from omegaconf import DictConfig, OmegaConf
import pandas as pd

def clean_dataset(config: DatasetDict, dataset: DatasetDict, set_name: str) -> DatasetDict: 
    # remove columns
    exclude_list = list(config.dataset[set_name].columns_to_remove)
    dataset = dataset.remove_columns(exclude_list)
    # rename columns
    src_col_list = list(config.dataset[set_name].original_column_names)
    new_col_list = list(config.dataset[set_name].new_column_names)
    for src_col, new_col in zip(src_col_list, new_col_list):
        dataset = dataset.rename_column(src_col, new_col)
    # return cleaned dataset
    return dataset

def prepare_and_create_dataloader(dataset: DatasetDict, lang: str, set_name: str, split: str, config: DictConfig, tokenizer: AutoTokenizer) -> DataLoader:
    # tokenize dataset
    if set_name == "pkavumba/balanced-copa":
        pass
    elif set_name == "xcorpa":
        translated_question = pd.read_csv("translation.csv")
        question_list = []
        for premise, question in zip(dataset["premise"], dataset["question"]):
            if question == "cause":
                sentence1 = premise + translated_question[translated_question["lang"] == lang]["cause"]
            elif question == "effect":
                sentence1 = premise + translated_question[translated_question["lang"] == lang]["effect"]
            question_list.append(sentence1)
        choice1_list = [choice1 for choice1 in dataset["choice1"]]
        choice2_list = [choice2 for choice2 in dataset["choice2"]]
        DatasetDict
        for 
        tokenized_set 

    # tokenize dataset
    col_to_tokenize = list(config.dataset[set_name].columns_to_tokenize)
    tokenized_set = tokenizer(dataset[split][[col_to_tokenize]], batched=True)
    # remove columns
    exclude_list = list(config.dataset[set_name].columns_to_remove)
    dataset = dataset.remove_columns(exclude_list)
    # rename columns
    src_col_list = list(config.dataset[set_name].original_column_names)
    new_col_list = list(config.dataset[set_name].new_column_names)
    for src_col, new_col in zip(src_col_list, new_col_list):
        dataset = dataset.rename_column(src_col, new_col)
    # set format
    tokenized_set.set_format("torch")
    # create dataloader
    dataloader = DataLoader(tokenized_set, shuffle=True, batch_size=config.params.batch_size, collate_fn=data_collator)
    return dataloader

def create_data_loaders(config: DictConfig, tokenizer: AutoTokenizer) -> tuple[list[DataLoader]]:
    # 
    def tokenize_function(example):
        return tokenizer(example["premise"], example["hypothesis"], truncation=True, padding=True)

    # load datasets
    dataset_dict = {}
    for set_cfg_name in config.dataset:
        set_languages = list(config.dataset[set_cfg_name].languages)
        dataset_name = config.dataset[set_cfg_name].name
        dataset = {lang : load_dataset(dataset_name, lang) for lang in set_languages}
        dataset_dict[set_cfg_name] = dataset

    train_loaders = []
    val_loaders = []
    test_loaders = []
    
    source_lang = config.params.source_lang
    for set_cfg_name, dataset in dataset_dict.items():
        for lang, dataset_in_lang in dataset.items():
            if lang == source_lang:
                # zero-shot cross lingual transfer
                # we only create train and val loaders for the source language
                train_split = config.dataset[set_cfg_name].train_split
                val_split = config.dataset[set_cfg_name].val_split
                train_loader = prepare_and_create_dataloader(dataset_in_lang, set_cfg_name, train_split, config, tokenize_function)
                val_loader = prepare_and_create_dataloader(dataset_in_lang, set_cfg_name, val_split, config, tokenize_function)
                train_loaders.append(train_loader)
                val_loaders.append(val_loader)
            else:
                # we only create test loaders for the target languages
                test_split = config.dataset[set_cfg_name].test_split
                test_loader = prepare_and_create_dataloader(dataset_in_lang, set_cfg_name, test_split, config, tokenize_function)
                test_loaders.append(test_loader)

    return train_loaders, val_loaders, test_loaders
                
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