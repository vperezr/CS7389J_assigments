import pandas as pd
from transformers import BertForSequenceClassification, BertConfig
from torch.utils.data import TensorDataset, random_split
from torch.optim import AdamW
from transformers import BertTokenizer
from torch.optim import AdamW

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import sys
import numpy as np
import time
import datetime

import pandas as pd
import torch
from transformers import BertTokenizer
from helpers import flat_accuracy


def tokenize_and_format(sentences):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    input_ids = []
    attention_masks = []
    for sent in sentences:
        encoded_dict = tokenizer(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 64,           # Pad & truncate all sentences.
                            padding = 'max_length',    # Use padding='max_length'
                            truncation = True,         # Explicitly set truncation
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                       )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    return input_ids, attention_masks

def flat_accuracy(preds, labels):
    pred_flat = torch.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return torch.sum(pred_flat == labels_flat)
