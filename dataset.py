from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from omegaconf import DictConfig

def load_datasets(cfg: DictConfig) -> dict[str, DatasetDict]:
    datasets_dict = {}
    for dataset in cfg.dataset:
        set_name = cfg._get_child("dataset").name
        set_languages = list(cfg._get_child("dataset").languages)
        dataset = {lang : load_dataset(set_name, lang) for lang in set_languages}
        datasets_dict[set_name] = dataset
    return datasets_dict

def tokenize_for_crosslingual_transfer(cfg: DictConfig, dataset: dict[str, DatasetDict], tokenizer: AutoTokenizer) -> dict[str, DataLoader]:
    source_language = cfg.params.source_language
    languages_lst = list(cfg.dataset.languages)
    batched = cfg.params.batched
    batch_size = cfg.params.batch_size

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    def tokenize_function(example):
        return tokenizer(example["premise"], example["hypothesis"], truncation=cfg.params.truncation, padding=cfg.params.padding)
    
    target_language_dataloaders = []
    for language in languages_lst:
        if language == source_language:
            tokenized_dataset = dataset[language].map(tokenize_function, batched=batched)
        else:
            tokenized_dataset = dataset[language]["validation"].map(tokenize_function, batched=batched)

        tokenized_dataset = tokenized_dataset.remove_columns(["premise", "hypothesis"])
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
        tokenized_dataset.set_format("torch")

        if language == source_language:
            source_language_train = tokenized_dataset["train"]
            source_language_eval = tokenized_dataset["validation"]
            train_dataloader = DataLoader(source_language_train, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
            eval_dataloader = DataLoader(source_language_eval, batch_size=batch_size, collate_fn=data_collator)
            source_language_dataloaders = [train_dataloader, eval_dataloader]
        else:
            target_dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=data_collator)

        target_language_dataloaders.append(target_dataloader)
    
    return source_language_dataloaders, target_language_dataloaders, languages_lst