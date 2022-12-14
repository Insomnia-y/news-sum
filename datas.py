import torch
from torch.utils.data import Dataset
import nlpaug.augmenter.word as naw
from nlpaug.flow import Sometimes
import re
WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))

class NewsDataset(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'r', encoding='utf-8') as f:
            train_data_all = f.readlines()
            for tem in train_data_all:
                tem = tem.split("\t")
                Data[int(tem[0])] = {
                    'title': tem[2][:-1],
                    'content': tem[1]
                }
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class NewsevalDataset(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'r', encoding='utf-8') as f:
            train_data_all = f.readlines()
            for tem in train_data_all:
                tem = tem.split("\t")
                Data[int(tem[0])] = {
                    'ID': tem[0],
                    'content': WHITESPACE_HANDLER(tem[1][:-1])
                }
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataevalCollator:
    def __init__(self, tokenizer,model,max_input_length=768,max_target_length=64):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.model = model
    def __call__(self, batch_samples):
        batch_inputs=[]
        for sample in batch_samples:
            batch_inputs.append(sample['content'])
        batch_data = self.tokenizer(
            batch_inputs,
            padding=True,
            max_length=self.max_input_length,
            truncation=True,
            return_tensors="pt"
        )
        return batch_data


class DataCollator:
    def __init__(self, tokenizer,model,max_input_length=512,max_target_length=64,training=True):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.model = model
        self.training = training
        self.aug = Sometimes([naw.SynonymAug(aug_p=0.1),  naw.RandomWordAug(aug_p=0.1)])
    def __call__(self, batch_samples):
        batch_inputs, batch_targets = [], []
        for sample in batch_samples:
            if self.training:
                batch_inputs.append(self.aug.augment(sample['content'], 1))
            else:
                batch_inputs.append(sample['content'])
            batch_targets.append(sample['title'])
        batch_data = self.tokenizer(
            batch_inputs,
            padding=True,
            max_length=self.max_input_length,
            truncation=True,
            return_tensors="pt"
        )
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                batch_targets,
                padding=True,
                max_length=self.max_target_length,
                truncation=True,
                return_tensors="pt"
            )["input_ids"]
            batch_data['decoder_input_ids'] = self.model.prepare_decoder_input_ids_from_labels(labels)
            end_token_index = torch.where(labels == self.tokenizer.eos_token_id)[1]
            for idx, end_idx in enumerate(end_token_index):
                labels[idx][end_idx+1:] = -100
            batch_data['labels'] = labels
        return batch_data

