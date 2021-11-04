import os
import pandas as pd
from spacy.training import Example
import random
from spacy.util import minibatch
import spacy
from spacy.pipeline import textcat
from tqdm import tqdm
import datetime

train_texts = list()
train_labels = list()
# add pos data
for f in os.listdir("../data/aclImdb/train/pos/"):
    file = open(f"../data/aclImdb/train/pos/{f}", 'r')
    data = file.read()

    train_texts.append(data)
    train_labels.append({'cats': {'pos': True,
                          'neg': False}})
    file.close()

# add neg data
for f in os.listdir("../data/aclImdb/train/neg/"):
    file = open(f"../data/aclImdb/train/neg/{f}", 'r')
    data = file.read()

    train_texts.append(data)
    train_labels.append({'cats': {'pos': False,
                          'neg': True}})
    file.close()

train_data = list(zip(train_texts, train_labels))
# Create an empty model and we will add pipe to it
nlp = spacy.blank("en") #新增空的NLP框架

#設定模型，這邊選用spaCy中預設的 text classfication 模型架構
config = {
   "threshold": 0.5,
   "model": textcat.DEFAULT_SINGLE_TEXTCAT_MODEL,
}
#將Text classfication 模型加入NLP框架中
textcat = nlp.add_pipe("textcat", config=config)
#設定需要辨識的label
textcat.add_label("pos")
textcat.add_label("neg")

# start training
optimizer = nlp.begin_training()

start = datetime.datetime.now()
for epoch in range(5):
    losses = {}
    random.shuffle(train_data)
    batches = minibatch(train_data, size=32) # Create the batch generator with batch size = 8

    for batch in tqdm(batches, desc=f"epoch {epoch}"):                   # Iterate through minibatches
        texts, labels = zip(*batch)         # use zip and unpack text, label for next line to update

        example = []
        for i in range(len(texts)):
            doc = nlp.make_doc(texts[i])
            #print(doc.text, labels[i])
            example.append(Example.from_dict(doc, labels[i]))
        nlp.update(example, sgd=optimizer, losses=losses)
    #nlp.update(example, drop=0.5, losses=losses)
    print(losses)
print(f"cost {datetime.datetime.now() - start} for training")

# 1~5 neg ;6~10 pos
texts = ["While Free Guy does a good job in the technical department, the overall storyline is bogged down by largely unfunny jokes.", 
        "Reynolds' old-fashioned jokes are not enough to prevent the game over of this dull video game proposal.",
        "Free Guy is not the movie it thinks it is. It is not a champion for originality. In fact, tag-teaming on punchlines with two of Hollywood's biggest franchises, it's hard to see it as nothing more than a corporate red herring in that regard.",
        "The movie is bursting with good vibes, but they get crushed by the climax: a bland brawl that bluntly reminds viewers they're watching a Disney product.", 
        "While I usually enjoy Ryan Reynolds, he annoyed the heck out of me in this one.",
        
        "The entertaining ensemble cast help create a popcorn-munching good time-in my case, the best time I've had at the movies all summer!",
        "Free Guy is a bright and colorful, action-packed family movie with great laughs and even a couple of surprise cameos which will surely raise a smile.", 
        "Free Guy is admittedly derivative, but it also has a huge heart, a lot of soul and a sneakily sly, ingenious, and subversive plot which belies its' day-glow, confectionary, high-gloss finish.",
        "If nothing else, Free Guy contains what might remain the best movie quip of the year.",
        "Highly enjoyable and wonderfully funny, with almost as much heart as high-octane special FX sequences."

]

docs = [nlp.tokenizer(text) for text in texts]
# Use textcat to get the scores for each doc
textcat = nlp.get_pipe('textcat')
scores = textcat.predict(docs)

print(scores)

# From the scores, find the label with the highest score/probability
predicted_labels = scores.argmax(axis=1)
print([textcat.labels[label] for label in predicted_labels])