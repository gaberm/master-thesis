from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from omegaconf import DictConfig, OmegaConf

class DataLoaderContainer:
    def __init__(self, name: str, language: str, split: str, loader: DataLoader):
        self.name = name
        self.language = language
        self.split = split
        self.loader = loader

def load_datasets(cfg: DictConfig) -> dict[str, DatasetDict]:
    datasets_dict = {}
    for ds_name in cfg.dataset:
        set_languages = list(cfg.dataset[ds_name].languages)
        dataset = {lang : load_dataset(ds_name, lang) for lang in set_languages}
        datasets_dict[ds_name] = dataset
    return datasets_dict

def tokenize_for_crosslingual_transfer(cfg: DictConfig, datasets: dict[str, DatasetDict], tokenizer: AutoTokenizer) -> list[DataLoaderContainer]:
    source_language = cfg.params.source_lang
    for ds_name in datasets.keys():
        languages_lst = cfg.dataset[ds_name].languages
        batched = cfg.params.batched
        batch_size = cfg.params.batch_size
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        def tokenize_function(example):
            return tokenizer(example["premise"], example["hypothesis"], truncation=cfg.params.truncation, padding=cfg.params.padding)
        
        data_loaders = []
        for language in languages_lst:
            if language == source_language:
                split = cfg.dataset[ds_name].source_lang_split
            else:
                split = cfg.dataset[ds_name].target_lang_split
            for s in split:
                tokenized_dataset = datasets[ds_name][language][s].map(tokenize_function, batched=batched)
                columns_to_remove = list(cfg.dataset[ds_name].columns_to_remove)
                tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)
                original_column_names = list(cfg.dataset[ds_name].original_column_names)
                new_column_names = list(cfg.dataset[ds_name].new_column_names)
                for og_name, new_name in zip(original_column_names, new_column_names):
                    tokenized_dataset = tokenized_dataset.rename_column(og_name, new_name)
                tokenized_dataset.set_format("torch")
                dataloader = DataLoader(tokenized_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
                loader = DataLoaderContainer(ds_name, language, s, dataloader)
                data_loaders.append(loader)
        return data_loaders