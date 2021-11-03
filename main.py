import os
import pandas as pd
from spacy.training import Example
import random
from spacy.util import minibatch
import spacy
from spacy.pipeline import textcat

train_data = list()
for f in os.listdir("../data/aclImdb/train/pos/"):
    file = open("../data/aclImdb/train/pos/{f}", 'r')
    data = [f.read()]

    train_data.append(zip(data, "pos"))
    print(train_data)
    file.close()