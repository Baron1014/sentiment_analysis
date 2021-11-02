import sys
import pandas as pd
from spacy.training import Example
import random
from spacy.util import minibatch
import spacy
from spacy.pipeline import textcat

spam = pd.read_csv("../data/spam.csv")
print(spam.head(10))

train_texts = spam['text'].values

# get each label from sapm['label'] and make a dic for indicating the ham and spam boolean value
train_labels = [{'cats': {'ham': label == 'ham',
                          'spam': label == 'spam'}} for label in spam['label']]

#將文本和label連結
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
textcat.add_label("ham")
textcat.add_label("spam")

# start training
optimizer = nlp.begin_training()

for epoch in range(5):
    losses = {}
    random.shuffle(train_data)
    batches = minibatch(train_data, size=8) # Create the batch generator with batch size = 8

    for batch in batches:                   # Iterate through minibatches
        texts, labels = zip(*batch)         # use zip and unpack text, label for next line to update

        example = []
        for i in range(len(texts)):
            doc = nlp.make_doc(texts[i])
            #
            #print(doc.text, labels[i])
            example.append(Example.from_dict(doc, labels[i]))
        nlp.update(example, sgd=optimizer, losses=losses)
    #nlp.update(example, drop=0.5, losses=losses)
    print(losses)


texts = ["Are you ready for the tea party????? It's gonna be wild",
         "URGENT Reply to this message for GUARANTEED FREE TEA",
         "time to vote, do you want to join us?",
         "ASAP to visit by tmr, Ben Hu",
         "This is a commercial advertisement message.",
         "XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here>> http://wap. xxxmobilemovieclub.com?n=QJKGIGHJJGCBL",
         'December only! Had your mobile 11mths+? You are entitled to update to the latest colour camera mobile for Free! Call The Mobile Update Co FREE on 08002986906'
         ]

docs = [nlp.tokenizer(text) for text in texts]
# Use textcat to get the scores for each doc
textcat = nlp.get_pipe('textcat')
scores = textcat.predict(docs)

print(scores)

# From the scores, find the label with the highest score/probability
predicted_labels = scores.argmax(axis=1)
print([textcat.labels[label] for label in predicted_labels])