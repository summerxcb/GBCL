from transformers import BertModel, BertTokenizer
import torch
import os
import re
import string
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import numpy as np
import langid
from scipy.spatial.distance import cosine
import networkx as nx
import torch.nn as nn
from torch_geometric.nn import GCNConv
import math
import torch.nn.functional as F
from sklearn.metrics import  precision_score, recall_score,accuracy_score,f1_score
# from Model import model
import torch.optim as optim
from torch_geometric.data import Data
from sklearn.metrics import classification_report
import warnings
import logging
device = torch.device("cuda")
pretrain_model_path = r"path/to/bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
bert_model = BertModel.from_pretrained(pretrain_model_path)
train_file_path = os.path.join(r"path/to/train_file_path")
validation_file_path = os.path.join(r"path/to/validation_file_path")
batch_size = 16
epochs = 10
#model
lstm_hidden_size = 768
gcn_input_size = 768
gcn_hidden_size = 128
hidden_dim=50