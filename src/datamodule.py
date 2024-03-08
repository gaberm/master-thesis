import platform
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from .model import load_tokenizer
from .dataset import prepare_copa, prepare_siqa, prepare_xcopa, prepare_paws_x, prepare_xnli, tokenize_ds, load_ds, download_ds

class CopaDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.data_dir = config.data_dir[platform.system().lower()]
        self.batch_size = config.params.batch_size
        self.tokenizer = load_tokenizer(config)
        self.data_collator = DataCollatorWithPadding(self.tokenizer)

    def prepare_data(self):
        # we train on balanced-copa and social_i_qa for copa
        download_ds("social_i_qa", "en", "train", self.data_dir)
        download_ds("social_i_qa", "en", "validation", self.data_dir)
        
        # we download balanced-copa, because copa is not available on HuggingFace
        # we ignore the mirrored rows from the balanced-copa dataset
        download_ds("pkavumba/balanced-copa", "en", "train", self.data_dir)
        download_ds("pkavumba/balanced-copa", "en", "test", self.data_dir)

    def setup(self, stage: str):
        if stage == "fit":
            self.train_data = [
                load_ds("pkavumba/balanced-copa", "en", "train", self.data_dir),
                load_ds("social_i_qa", "en", "train", self.data_dir)
            ]
            self.train_data = [
                prepare_copa(self.train_data[0]),
                prepare_siqa(self.train_data[1])
            ]
            self.train_data = [
                tokenize_ds(self.train_data[0], self.tokenizer),
                tokenize_ds(self.train_data[1], self.tokenizer)
            ]
            
            self.val_data = [
                load_ds("pkavumba/balanced-copa", "en", "test", self.data_dir),
                load_ds("social_i_qa", "en", "validation", self.data_dir)
            ]
            self.val_data = [
                prepare_copa(self.val_data[0]),
                prepare_siqa(self.val_data[1])
            ]
            self.val_data = [
                tokenize_ds(self.val_data[0], self.tokenizer),
                tokenize_ds(self.val_data[1], self.tokenizer)
            ]
    
    def train_dataloader(self):
        train_dataloader = [DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            )
            for dataset in self.train_data]
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = [DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.data_collator
            ) 
            for dataset in self.val_data]
        return val_dataloader
    

class XcopaDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.data_dir = config.data_dir[platform.system().lower()]
        self.batch_size = config.params.batch_size
        self.tokenizer = config.tokenizer
        self.tokenizer = load_tokenizer(config)
        self.data_collator = DataCollatorWithPadding(self.tokenizer)
        self.lang = config.dataset.lang

    def prepare_data(self):
        for lang in self.lang:
            download_ds("xcopa", lang, "test", self.data_dir)

    def setup(self, stage: str):
        if stage == "test":
            self.test_data = [load_ds("xcopa", lang, "test", self.data_dir) for lang in self.lang]
            self.test_data = [prepare_xcopa(ds) for ds in self.test_data]
            self.test_data = [tokenize_ds(ds, self.tokenizer) for ds in self.test_data]
    
    def test_dataloader(self):
        test_dataloader_lst = []
        for ds in self.test_data:
            test_dataloader = DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self.data_collator,
            )
            test_dataloader_lst.append(test_dataloader)
        return test_dataloader_lst
    

class PawsXDataModule(LightningDataModule):
    def __init__(self, config):    
        super().__init__()
        self.data_dir = config.data_dir[platform.system().lower()]
        self.batch_size = config.params.batch_size
        self.tokenizer = load_tokenizer(config)
        self.data_collator = DataCollatorWithPadding(self.tokenizer)
        self.lang = config.dataset.lang

    def prepare_data(self):
        download_ds("paws-x", "en", "train", self.data_dir)
        download_ds("paws-x", "en", "validation", self.data_dir)
        for lang in self.lang:
            download_ds("paws-x", lang, "test", self.data_dir)

    def setup(self, stage: str):
        if stage == "fit":
            self.train_data = load_ds("paws-x", "en", "train", self.data_dir)
            self.train_data = prepare_paws_x(self.train_data)
            self.train_data = tokenize_ds(self.train_data, self.tokenizer)

            self.val_data = load_ds("paws-x", "en", "validation", self.data_dir)
            self.val_data = prepare_paws_x(self.val_data)
            self.val_data = tokenize_ds(self.val_data, self.tokenizer)
        
        if stage == "test":
            self.test_data = [load_ds("paws-x", lang, "test", self.data_dir) for lang in self.lang]
            self.test_data = [prepare_paws_x(ds) for ds in self.test_data]
            self.test_data = [tokenize_ds(ds, self.tokenizer) for ds in self.test_data]

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
        )
        return train_dataloader
    
    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
        )
        return val_dataloader
    
    def test_dataloader(self):
        test_dataloader_lst = []
        for ds in self.test_data:
            test_dataloader = DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=self.data_collator,
            )
            test_dataloader_lst.append(test_dataloader)
        return test_dataloader_lst
    

class XnliDataModule(LightningDataModule):
    def __init__(self, config):    
        super().__init__()
        self.data_dir = config.data_dir[platform.system().lower()]
        self.batch_size = config.params.batch_size
        self.tokenizer = load_tokenizer(config)
        self.data_collator = DataCollatorWithPadding(self.tokenizer)
        self.lang = config.dataset.lang

    def prepare_data(self):
        download_ds("xnli", "en", "train", self.data_dir)
        download_ds("xnli", "en", "validation", self.data_dir)
        for lang in self.lang:
            download_ds("xnli", lang, "test", self.data_dir)

    def setup(self, stage: str):
        if stage == "fit":
            self.train_data = load_ds("xnli", "en", "train", self.data_dir)
            self.train_data = prepare_xnli(self.train_data)
            self.train_data = tokenize_ds(self.train_data, self.tokenizer)

            self.val_data = load_ds("xnli", "en", "validation", self.data_dir)
            self.val_data = prepare_xnli(self.val_data)
            self.val_data = tokenize_ds(self.val_data, self.tokenizer)
        
        if stage == "test":
            self.test_data = [load_ds("xnli", lang, "test", self.data_dir) for lang in self.lang] 
            self.test_data = [prepare_xnli(ds) for ds in self.test_data]
            self.test_data = [tokenize_ds(ds, self.tokenizer) for ds in self.test_data]

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
        )
        return train_dataloader
    
    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
        )
        return val_dataloader
    
    def test_dataloader(self):
        test_dataloader_lst = []
        for ds in self.test_data:
            test_dataloader = DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=self.data_collator,
            )
            test_dataloader_lst.append(test_dataloader)
        return test_dataloader_lst


def create_data_modules(config):
    data_modules = {
        "copa": CopaDataModule,
        "xcopa": XcopaDataModule,
        "paws_x": PawsXDataModule,
        "xnli": XnliDataModule
    }
    return data_modules[config.dataset.name](config)
        

