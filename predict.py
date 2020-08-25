import csv
import tweepy
import ssl
import pandas as pd
import numpy as np
from textblob import TextBlob



name = 'realDonaldTrump'
tweet_id = '1296638316281044992'
consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
ssl._create_default_https_context = ssl._create_unverified_context
api = tweepy.API(auth)
api = tweepy.API(auth, wait_on_rate_limit=True)

#donald

replies=[]
for tweet in tweepy.Cursor(api.search,q='to:'+name, result_type='recent', timeout=999999).items(2000):
    if hasattr(tweet, 'in_reply_to_status_id_str'):
        if (tweet.in_reply_to_status_id_str==tweet_id):
            replies.append(tweet)

with open('trump.csv', 'a+') as f:
    csv_writer = csv.DictWriter(f, fieldnames=('user', 'text'))
    csv_writer.writeheader()
    for tweet in replies:
        row = {'user': tweet.user.screen_name, 'text': tweet.text.replace('\n', ' ')}
        csv_writer.writerow(row)

name ='JoeBiden'
tweet_id='1297570544867475458'

replies=[]
for tweet in tweepy.Cursor(api.search,q='to:'+name, result_type='recent', timeout=999999).items(2000):
    if hasattr(tweet, 'in_reply_to_status_id_str'):
        if (tweet.in_reply_to_status_id_str==tweet_id):
            replies.append(tweet)

#joe

with open('biden.csv', 'a+') as f:
    csv_writer = csv.DictWriter(f, fieldnames=('user', 'text'))
    csv_writer.writeheader()
    for tweet in replies:
        row = {'user': tweet.user.screen_name, 'text': tweet.text.replace('\n', ' ')}
        csv_writer.writerow(row)

trump_review = pd.read_csv('trump.csv', encoding='utf-8')
biden_review = pd.read_csv('biden.csv', encoding='utf-8')

trump_review.head()
biden_review.head()

trump_review['Sentiment_Polarity'] = trump_review['text'].apply(find_pol)
trump_review.tail()


def find_pol(review):
    return TextBlob(review).sentiment.polarity


biden_review['Sentiment_Polarity'] = biden_review['text'].apply(find_pol)
biden_review.tail()

trump_review['Expression Label'] = np.where(trump_review['Sentiment_Polarity'] > 0, 'positive', 'negative')
trump_review['Expression Label'][trump_review.Sentiment_Polarity == 0] = "Neutral"
trump_review.tail()

biden_review['Expression Label'] = np.where(biden_review['Sentiment_Polarity'] > 0, 'positive', 'negative')
biden_review['Expression Label'][biden_review.Sentiment_Polarity == 0] = "Neutral"
biden_review.tail()

new1 = trump_review.groupby('Expression Label').count()
x = list(new1['Sentiment_Polarity'])
y = list(new1.index)
tuple_list = list(zip(x, y))
df = pd.DataFrame(tuple_list, columns=['x', 'y'])
df['color'] = 'yellow'
df['color'][1] = 'green'
df['color'][2] = 'blue'
fig = go.Figure(go.Bar(x=df['x'],
                       y=df['y'],
                       orientation='h',
                       marker={'color': df['color']}))
fig.update_layout(title_text='Trump\'s Reviews Analysis')
fig.show()

new2 = biden_review.groupby('Expression Label').count()
x = list(new2['Sentiment_Polarity'])
y = list(new2.index)
tuple_list = list(zip(x, y))
df = pd.DataFrame(tuple_list, columns=['x', 'y'])
df['color'] = 'yellow'
df['color'][1] = 'green'
df['color'][2] = 'blue'
fig = go.Figure(go.Bar(x=df['x'],
                       y=df['y'],
                       orientation='h',
                       marker={'color': df['color']}))
fig.update_layout(title_text='Biden\'s Reviews Analysis')
fig.show()

reviews1 = trump_review[trump_review['Sentiment_Polarity'] == 0.0000]
reviews1.shape
cond1 = trump_review['Sentiment_Polarity'].isin(reviews1['Sentiment_Polarity'])
trump_review.drop(trump_review[cond1].index, inplace=True)
trump_review.shape

reviews2 = biden_review[biden_review['Sentiment_Polarity'] == 0.0000]
reviews2.shape
cond2 = biden_review['Sentiment_Polarity'].isin(reviews1['Sentiment_Polarity'])
biden_review.drop(biden_review[cond2].index, inplace=True)
biden_review.shape

np.random.seed(10)
remove_n = 324
drop_indices = np.random.choice(trump_review.index, remove_n)
df_subset_trump = trump_review.drop(drop_indices)
df_subset_trump.shape

np.random.seed(10)
remove_n = 31
drop_indices = np.random.choice(biden_review.index, remove_n)
df_subset_biden = biden_review.drop(drop_indices)
df_subset_biden.shape