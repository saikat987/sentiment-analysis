import twitter

# initialize api instance
twitter_api = twitter.Api(consumer_key='consumer key',
                        consumer_secret='consumer secret',
                        access_token_key='###############',
                        access_token_secret='#######################')

# test authentication
print(twitter_api.VerifyCredentials())
#downloading test dataset

def buildTestSet(search_keyword):
    try:
        tweets_fetched = twitter_api.GetSearch(search_keyword, count = 100)
        
        print("Fetched " + str(len(tweets_fetched)) + " tweets for the term " + search_keyword)
        
        return [{"text":status.text, "label":None} for status in tweets_fetched]
    except:
        print("Unfortunately, something went wrong..")
        return None
search_term = input("Enter a search keyword:")
testDataSet = buildTestSet(search_term)
print(testDataSet[0:4])

#function to download train dataset

def buidTrainingSet(corpusFile, tweetDataFile):
    import csv
    import time
    
    corpus = []
    
    with open(corpusFile,'r') as csvfile:
        lineReader = csv.reader(csvfile,delimiter=',', quotechar="\"")
        for row in lineReader:
            corpus.append({"tweet_id":row[2], "label":row[1], "topic":row[0]})
    
    rate_limit = 180
    sleep_time = 900/180
    
    trainingDataSet = []
    
    for tweet in corpus:
        try:
            status = twitter_api.GetStatus(tweet["tweet_id"])
            print("Tweet fetched" + status.text)
            tweet["text"] = status.text
            trainingDataSet.append(tweet)
            #since we can do 180 request in 15 minutes we are stalling the loop for a while 
            time.sleep(sleep_time) 
        except: 
            continue
    # now we write them to the empty CSV file
    with open(tweetDataFile,'wb') as csvfile:
        linewriter = csv.writer(csvfile,delimiter=',',quotechar="\"")
        for tweet in trainingDataSet:
            try:
                linewriter.writerow([tweet["tweet_id"], tweet["text"], tweet["label"], tweet["topic"]])
            except Exception as e:
                print(e)
    return trainingDataSet
corpusFile = "corpus.csv"
tweetDataFile = "tweetDataFile.csv"
#downloading the dataset (note:already downloaded this will take more than 6 hour )
trainingData = buidTrainingSet(corpusFile, tweetDataFile)


#saving the traning dataset raw

with open('training.tsv', 'w', newline='', encoding="utf-8") as f_output:
    tsv_output = csv.writer(f_output, delimiter='\t')
    tsv_output.writerow(['text','label'])
    for i in trainingData:
        tsv_output.writerow([i['text'],i['label']])
#cleaning the downloaded data
import re
from nltk.tokenize import word_tokenize
from string import punctuation 
from nltk.corpus import stopwords 

class PreProcessTweets:
    def __init__(self):
        self._stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL'])
        
    def processTweets(self, list_of_tweets):
        processedTweets=[]
        for tweet in list_of_tweets:
            processedTweets.append((self._processTweet(tweet["text"]),tweet["label"]))
        return processedTweets
    
    def _processTweet(self, tweet):
        tweet = tweet.lower() # convert text to lower-case
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet) # remove URLs
        tweet = re.sub('@[^\s]+', 'AT_USER', tweet) # remove usernames
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag
        tweet = word_tokenize(tweet) # remove repeated characters (helloooooooo into hello)
        tweet=' '.join(tweet)
        return tweet

tweetProcessor = PreProcessTweets()
preprocessedTrainingSet = tweetProcessor.processTweets(trainingData)
preprocessedTestSet = tweetProcessor.processTweets(testDataSet)



import csv
import pandas as pd

#creating train .csv from trainDataSet  
with open('train.tsv', 'w', newline='', encoding="utf-8") as f_output:
    tsv_output = csv.writer(f_output, delimiter='\t')
    tsv_output.writerow(['text','label'])
    for i in preprocessedTrainingSet:
        tsv_output.writerow([i[0],i[1]])
  
 
#creating test.tsv from testDataSet
with open('test.tsv', 'w', newline='', encoding="utf-8") as f_output:
    tsv_output = csv.writer(f_output, delimiter='\t')
    tsv_output.writerow(['text','label'])
    for i in testDataSet:
        tsv_output.writerow([i['text']])
#reading the train dataset
dataset=pd.read_csv("train.tsv",delimiter='\t',quoting=3)
testdataset=pd.read_csv("test.tsv",delimiter='\t',quoting=3)

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, len(dataset['text'])):
    review = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
corpus1=[]
for i in range(0, len(testdataset['text'])):
    review = re.sub('[^a-zA-Z]', ' ', testdataset['text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values
X_train=X[:4114]
X_test=X[4114:]
#training the dataset
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y)

#predicting the sentiments
y_pred = classifier.predict(X_test)
        
    
        

