import pandas as pd
from spacy.training import Example
import random
from spacy.util import minibatch
import spacy
from spacy.pipeline.textcat import DEFAULT_SINGLE_TEXTCAT_MODE

movie = pd.read_csv("../data")