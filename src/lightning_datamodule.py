from lightning import LightningDataModule
from datasets import load_dataset, load_from_disk, Dataset, concatenate_datasets
from torch.utils.data import DataLoader
from dataset import prepare_copa, prepare_siqa, tokenize_dataset

class CopaDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.train_data = load_dataset("pkavumba/balanced-copa", "en", split="train")
        self.val_data = load_dataset("pkavumba/balanced-copa", "en", split="test")
        self.batch_size = config.params.batch_size

    def setup(self):
        self.train_data = prepare_copa(self.train_data)
        self.val_data = prepare_copa(self.val_data)

        self.train_data = tokenize_dataset(self.train_data, self.tokenizer)
        self.val_data = tokenize_dataset(self.val_data, self.tokenizer)
    
    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=False,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
        )
        return val_dataloader
    

class SiqaDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.train_data = load_dataset("social_i_qa", split="train")
        self.val_data = load_dataset("social_i_qa", split="validation")
        self.batch_size = config.params.batch_size
        self.tokenizer = config.tokenizer

    def setup(self):
        self.train_data = prepare_siqa(self.train_data)
        self.val_data = prepare_siqa(self.val_data)

        self.train_data = tokenize_dataset(self.train_data, self.tokenizer)
        self.val_data = tokenize_dataset(self.val_data, self.tokenizer)
    
    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=False,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
        )
        return val_dataloader
    

class XcopaDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.train_data = load_dataset("social_i_qa", split="train")
        self.val_data = load_dataset("social_i_qa", split="validation")
        self.batch_size = config.params.batch_size
        self.tokenizer = config.tokenizer

    def setup(self):
        self.train_data = prepare_siqa(self.train_data)
        self.val_data = prepare_siqa(self.val_data)

        self.train_data = tokenize_dataset(self.train_data, self.tokenizer)
        self.val_data = tokenize_dataset(self.val_data, self.tokenizer)
    
    def test_dataloader(self):
        test_dataloader = DataLoader(
            self.test_loader,
            batch_size=self.batch_size,
            shuffle=False,
        )
        return test_dataloader





