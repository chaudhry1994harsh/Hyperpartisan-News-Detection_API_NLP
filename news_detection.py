from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import json



def tokenIDsale(inputarr): 
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    tokenized = inputarr.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

    for x in range(len(tokenized)):
        y = len(tokenized[x])
        if(y>512):
            start = tokenized[x][: 129]
            end = tokenized[x][y-383 :]
            temp = start + end
            tokenized[x] = temp

    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)

    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

    return padded


'''
article = pd.read_csv('data files/articles-training-byarticle.csv')
content = tokenIDsale(article['content']) 
contentLABEL = article['truth']

optimizer = tf.keras.optimizers.Adam(learning_rate=0.000001)    
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model = TFDistilBertForSequenceClassification.from_pretrained("data files/article Model ACC/")

model.compile(optimizer=optimizer, loss=loss,metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])
result = model(np.array([content[0],]))
_,test_acc = model.evaluate(x = np.array([content[15],]), y = np.array([contentLABEL[15],]))
_,test_acc = model.evaluate(x = content, y = contentLABEL)
'''

def detect(content):
    content = pd.read_json(content)
    content = content['articles'] 
    content = tokenIDsale(content)
    response = []

    model = TFDistilBertForSequenceClassification.from_pretrained("data files/article Model ACC/")

    for x in content:
        resp_temp ={}
        result = model(np.array([x,]))
        result = result[0][0]
        #right positive: true 
        #left positive: false
        if result[0] < result[1]: resp_temp['hyperpartisan detected'] = 'YES'
        elif result[0] > result[1]: resp_temp['hyperpartisan detected'] = 'NO'
        else: resp_temp['hyperpartisan detected'] = 'non conclusive'
        response.append(resp_temp)
    return json.dumps(response)
