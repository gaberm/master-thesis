import platform
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from .model import load_tokenizer
from .dataset import prepare_copa, prepare_siqa, prepare_xcopa, prepare_paws_x, prepare_xnli, tokenize_ds, load_ds

class CopaDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.data_dir = config.data_dir[platform.system().lower()]
        # we load balanced-copa, because copa is not available on HuggingFace
        self.train_data = load_ds("pkavumba/balanced-copa", "en", "train", self.data_dir)
        self.val_data = load_ds("pkavumba/balanced-copa", "en", "test", self.data_dir)
        self.batch_size = config.params.batch_size
        self.tokenizer = load_tokenizer(config)

    def setup(self):
        self.train_data = prepare_copa(self.train_data)
        self.val_data = prepare_copa(self.val_data)

        self.train_data = tokenize_ds(self.train_data, self.tokenizer)
        self.val_data = tokenize_ds(self.val_data, self.tokenizer)
    
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
        self.data_dir = config.data_dir[platform.system().lower()]
        self.train_data = load_ds("social_i_qa", "en", "train", self.data_dir)
        self.val_data = load_ds("social_i_qa", "en", "validation", self.data_dir)
        self.batch_size = config.params.batch_size
        self.tokenizer = load_tokenizer(config)

    def setup(self):
        self.train_data = prepare_siqa(self.train_data)
        self.val_data = prepare_siqa(self.val_data)

        self.train_data = tokenize_ds(self.train_data, self.tokenizer)
        self.val_data = tokenize_ds(self.val_data, self.tokenizer)
    
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
        self.data_dir = config.data_dir[platform.system().lower()]
        self.test_data = [load_ds("xcopa", lang, "test", self.data_dir) for lang in config.xcopa.langs]
        self.batch_size = config.params.batch_size
        self.tokenizer = config.tokenizer
        self.tokenizer = load_tokenizer(config)

    def setup(self):
        self.test_data = [prepare_xcopa(ds) for ds in self.test_data]
        self.test_data = [tokenize_ds(ds, self.tokenizer) for ds in self.test_data]
    
    def test_dataloader(self):
        test_dataloader_lst = []
        for ds in self.test_data:
            test_dataloader = DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
            )
            test_dataloader_lst.append(test_dataloader)
        return test_dataloader_lst
    

class PawsXDataModule(LightningDataModule):
    def __init__(self, config):    
        super().__init__()
        self.data_dir = config.data_dir[platform.system().lower()]
        self.train_data = load_ds("paws-x", "en", "train", self.data_dir)
        self.val_data = load_ds("paws-x", "en", "validation", self.data_dir)
        self.test_data = [load_ds("paws-x", lang, "test", self.data_dir) for lang in config.paws_x.langs]
        self.batch_size = config.params.batch_size
        self.tokenizer = load_tokenizer(config)

    def setup(self):
        self.train_data = prepare_paws_x(self.train_data)
        self.val_data = prepare_paws_x(self.val_data)
        self.test_data = [prepare_paws_x(ds) for ds in self.test_data]

        self.train_data = tokenize_ds(self.train_data, self.tokenizer)
        self.val_data = tokenize_ds(self.val_data, self.tokenizer)
        self.test_data = [tokenize_ds(ds, self.tokenizer) for ds in self.test_data]

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
        )
        return train_dataloader
    
    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=True,
        )
        return val_dataloader
    
    def test_dataloader(self):
        test_dataloader_lst = []
        for ds in self.test_data:
            test_dataloader = DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=True,
            )
            test_dataloader_lst.append(test_dataloader)
        return test_dataloader_lst
    

class XnliDataModule(LightningDataModule):
    def __init__(self, config):    
        super().__init__()
        self.data_dir = config.data_dir[platform.system().lower()]
        self.train_data = load_ds("xnli", "en", "train", self.data_dir)
        self.val_data = load_ds("xnli", "en", "validation", self.data_dir)
        self.test_data = [load_ds("xnli", lang, "test", self.data_dir) for lang in config.xnli.langs]
        self.batch_size = config.params.batch_size
        self.tokenizer = load_tokenizer(config)

    def setup(self):
        self.train_data = prepare_xnli(self.train_data)
        self.val_data = prepare_xnli(self.val_data)
        self.test_data = [prepare_xnli(ds) for ds in self.test_data]

        self.train_data = tokenize_ds(self.train_data, self.tokenizer)
        self.val_data = tokenize_ds(self.val_data, self.tokenizer)
        self.test_data = [tokenize_ds(ds, self.tokenizer) for ds in self.test_data]

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
        )
        return train_dataloader
    
    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=True,
        )
        return val_dataloader
    
    def test_dataloader(self):
        test_dataloader_lst = []
        for ds in self.test_data:
            test_dataloader = DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=True,
            )
            test_dataloader_lst.append(test_dataloader)
        return test_dataloader_lst


def load_train_val_ds(config):
    data_modules = {
        "copa": CopaDataModule,
        "siqa": SiqaDataModule,
        "xcopa": XcopaDataModule,
        "paws_x": PawsXDataModule,
        "xnli": XnliDataModule
    }

    train_module_lst = []
    val_module_lst = []
    for ds in config.dataset:
        train_module_lst.append(data_modules[ds.name](config).train_dataloader())
        val_module_lst.append(data_modules[ds.name](config).val_dataloader())

    return train_module_lst, val_module_lst


def load_test_ds(config):
    data_modules = {
        "xcopa": XcopaDataModule,
        "paws_x": PawsXDataModule,
        "xnli": XnliDataModule
    }

    return data_modules[config.dataset[0].name](config).test_dataloader()


