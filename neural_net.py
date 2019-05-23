import csv
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import json
import keras as k

f = open('data/2018/10/01/00/29.json', 'r')
lines = f.readlines()

tweets = []
labels = []

with open('data/russian_tweets/IRAhandle_tweets_1.csv', newline='') as csvfile:
    categories = csvfile.readline().split(",")
    tweetreader = csv.reader(csvfile, delimiter=',')
    for row in tweetreader:
        tweet = dict(zip(categories, row))
        if tweet['language'] == 'English':
            tweets.append(tweet['content'])
            labels.append(1)

num_bad = len(tweets)

for line in lines:
    tweet = json.loads(line)
    if 'user' in tweet.keys():
        if tweet["user"]["lang"] == "en":
            tweets.append(tweet['text'])
            labels.append(0)

vectorizer = CountVectorizer(binary=True, lowercase=True)

X = vectorizer.fit_transform(np.array(tweets))

#TO BE CONTINUED
model = k.models.Sequential();
model.add(k.layers.Dense(len(vectorizer.get_feature_names()), activation='softmax'))
model.add(k.layers.Dense(len(vectorizer.get_feature_names()), activation='softmax'))
model.add(k.layers.Dense(len(vectorizer.get_feature_names()), activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])













