import pandas as pd
from spacy.training import Example
import random
from spacy.util import minibatch
import spacy
from spacy.pipeline import textcat

movie = pd.read_csv("../data")