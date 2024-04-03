import torch
import datasets

from typing import List, Dict
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def load_data(path: str=r'./nlpcc2017_clean.json',file_type: str=r'json'):

    def _flatten(example):
        return {
            "document": example["content"],
            "summary": example["title"],
            "id":"0"
        }

    dataset = load_dataset(file_type, 
                           data_files=path, 
                           field='data')
    dataset = dataset["train"].map(_flatten, 
                                   remove_columns=["title", "content"])
    train_dataset, valid_dataset = dataset.train_test_split(
                                    test_size=0.1,
                                    shuffle=True,
                                    seed=42).values()
    train_dataset, test_dataset = train_dataset.train_test_split(
                                    test_size=0.1,
                                    shuffle=True,
                                    seed=42).values()
    data_sets = datasets.DatasetDict({
                    "train"      : train_dataset,
                    "validation" : valid_dataset,
                    "test"       : test_dataset
                    })
    return data_sets



def tokenized(datasets, tokenizer, max_input_length, max_target_length):

    def _preprocess(examples):
        """
        @Usages:
            tokenized_datasets = tokenized_datasets.map(preprocess, 
                                    batched=True, 
                                    remove_columns=["document", "summary", "id"])
        @Return: 
            train: Dataset({
            features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
            num_rows: 40454
        })
        """
        inputs = [doc for doc in examples["document"]]
        model_inputs = tokenizer(inputs, 
                                max_length=max_input_length, 
                                truncation=True) # truncation 表示对于过长的部分进行截断

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["summary"], 
                            max_length=max_target_length, 
                            truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    
    tokenized_datasets = datasets.map(_preprocess, 
                                      batched=True, 
                                      remove_columns=["document", "summary", "id"])
    return tokenized_datasets



def collate_fn(features: Dict) -> Dict:
    """
    @arg1: features
            ['input_ids', 
            'token_type_ids', 
            'attention_mask', 
            'labels']
    @function: 将features转换成张量，并且进行填充对齐。
            方便后续的批量处理。

    """
    batch_input_ids = [torch.LongTensor(feature["input_ids"]) 
                        for feature in features]
    batch_attention_mask = [torch.LongTensor(feature["attention_mask"]) 
                        for feature in features]
    batch_labels = [torch.LongTensor(feature["labels"]) 
                    for feature in features]
    
    # padding
    batch_input_ids = pad_sequence(batch_input_ids, 
                                   batch_first=True, 
                                   padding_value=0)
    batch_attention_mask = pad_sequence(batch_attention_mask, 
                                        batch_first=True, 
                                        padding_value=0)
    batch_labels = pad_sequence(batch_labels, 
                                batch_first=True, 
                                padding_value=-100)
    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels
        }



