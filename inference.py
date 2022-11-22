MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
input_file_name = '../data/test_young_adult.csv'

from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import pandas as pd
import tqdm
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)



tokenizer = AutoTokenizer.from_pretrained(MODEL, model_max_length=512)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def get_preds(text):
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    # Print labels and scores
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    l = config.id2label[ranking[0]]
    s = scores[ranking[0]]
    return l,s

df = pd.read_csv(input_file_name)
df = shuffle(df).iloc[:1000]
texts = df['text']
sentiment = []
scores = []
for text in tqdm.tqdm(texts):
    l, s = get_preds(text)
    sentiment.append(l)
    scores.append(s)

df['Sentiment'] = sentiment
df['Score'] = scores

hist = df.Sentiment.value_counts()[1:].plot(kind='bar')
plt.savefig(input_file_name.split('/')[-1].split('.')[0] + '.png')
df.to_csv(input_file_name.split('/')[-1].split('.')[0] + '_predictions.csv')