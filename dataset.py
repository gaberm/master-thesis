from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from omegaconf import DictConfig

def load_datasets(cfg: DictConfig) -> dict[str, DatasetDict]:
    dataset = cfg.name
    dataset_languages = list(cfg.dataset.languages)
    train_datasets = {language : load_dataset(dataset, language) for language in dataset_languages}
    return train_datasets

def tokenize_function(cfg: DictConfig, tokenizer: AutoTokenizer, example: DatasetDict) -> AutoTokenizer:
    return tokenizer(example["premise"], example["hypothesis"], truncation=cfg.params.truncation, padding=cfg.params.padding)

def tokenize_dataset(cfg: DictConfig, dataset: dict[str, DatasetDict], tokenizer: AutoTokenizer) -> dict[str, DataLoader]:
    source_language = cfg.train.source_language
    dataset_languages = list(cfg.dataset.languages)
    batched = cfg.params.batched
    batch_size = cfg.params.batch_size
    tokenize_fct = tokenize_function
    
    dataloaders = {}
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    for language in dataset_languages:
        if language == source_language:
            tokenized_dataset = dataset[language].map(tokenize_function, batched=batched)
        else:
            tokenized_dataset = dataset[language]["validation"].map(tokenize_function, batched=batched)

        tokenized_dataset = tokenized_dataset.remove_columns(["premise", "hypothesis"])
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
        tokenized_dataset.set_format("torch")

        if language == source_language:
            en_train = tokenized_dataset["train"]
            train_dataloader = DataLoader(en_train, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
            eval_dataloader = DataLoader(tokenized_dataset["validation"], batch_size=batch_size, collate_fn=data_collator)
            dataloader = {"train" : train_dataloader, "validation" : eval_dataloader}
        else:
            dataloader = DataLoader(tokenized_dataset, batch_size=cfg.train.batch_size, collate_fn=data_collator)

        dataloaders[language] = dataloader
    
    return dataloaders