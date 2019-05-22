import csv
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import json

from sklearn.naive_bayes import GaussianNB

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

model = GaussianNB()

model.fit(X.toarray(), labels)

predicted_troll = model.predict(tweets[0])
print(predicted_troll)

predicted_actual = model.predict(tweets[num_bad + 100])
print(predicted_actual)











