import sys
def twitter_competitor_analysis_weekly(uri,mydb,tag):

	import pandas as pd
	import numpy as np
	import re
	import pymongo as pym
	import nltk
	import datetime
	import psycopg2
	from sqlalchemy import create_engine
	from nltk.corpus import stopwords
	from dateutil.relativedelta import relativedelta
	from configparser import ConfigParser
	import pytz
	IST = pytz.timezone('Asia/Kolkata')
	from nltk.stem.wordnet import WordNetLemmatizer
	from sklearn.feature_extraction.text import CountVectorizer
	from nltk.sentiment.vader import SentimentIntensityAnalyzer
	sid = SentimentIntensityAnalyzer()
	from datetime import datetime,timedelta
	import pytz
	IST = pytz.timezone('Asia/Kolkata')
	Today=pd.to_datetime(datetime.strftime(datetime.now(IST), '%Y-%m-%d'))
	yesterday=datetime.strftime(datetime.now(IST) - timedelta(1), '%Y-%m-%d')
	prior_7days=datetime.strftime(datetime.now(IST) - timedelta(7), '%Y-%m-%d')

	def fetch(uri,mydb):

		uri = uri #"mongodb://localhost:27017"
		client= pym.MongoClient(uri)
		mydb = client[mydb] #client['socialmedia']
		hashtag_tweets= mydb['competitor_hashtag_tweets']#hashtag_tweets#hashtag_tweets_songs
		df_dumped= pd.json_normalize(hashtag_tweets.find())

		return df_dumped

	def transform(df_dumped):

		df_dumped[["createdAt"]]=df_dumped[["createdAt"]].apply(pd.to_datetime, errors='coerce') 
		df_dumped=df_dumped.sort_values('createdAt')

		#Dropping the tweets with empty Id and empty CreatedAt
		df_dumped=df_dumped[~df_dumped['createdAt'].isnull()]
		df_dumped=df_dumped[df_dumped['createdAt']>=prior_7days]
		df_dumped=df_dumped[df_dumped['createdAt']<Today]
		df_dumped=df_dumped[~df_dumped['id'].isnull()]
		df_dumped['id']=df_dumped['id'].astype('str')
		df=df_dumped.drop_duplicates(['id'],keep='last')

		df['text']=df['text'].str.lower()
		df['text'] = df['text'].str.replace('[^A-Za-z0-9]', ' ', flags=re.UNICODE)

		if tag=='Vikram Vedha':
			df=df[df['text'].str.contains('vikram|vedha|hrithik|saif|radhika apte|radhikaapte|yogitabihani')]
		elif tag=='Naane Varuvean':
			df=df[df['text'].str.contains('naane|varuvean|varuven|dhanush|selvaraghavan|vcreations|veerasoora|veera soora')]
		elif tag=='Brahmastra':
			df=df[df['text'].str.contains('brahmastra|shiva|amitabh|ranbir|kapoor|alia|mouni roy|part one|karan|johar')]


		df['retweetedStatus.id'] = pd.to_numeric(df['retweetedStatus.id'], errors='coerce').convert_dtypes()
		df['original_tweet_id']=df['retweetedStatus.id']

		df1=df[~df['original_tweet_id'].isnull()]
		df2=df[df['original_tweet_id'].isnull()]
		df2['original_tweet_id']=df['id']
		df=pd.concat([df1,df2],ignore_index=True)

		df['inReplyToStatusId']=pd.to_numeric(df['inReplyToStatusId'], errors='coerce').convert_dtypes()
		df_reply=df[df['inReplyToStatusId']>0]
		df_not_reply=df[(df['inReplyToStatusId']<0) | (df['inReplyToStatusId'].isnull())] #because 'inReplyToStatusId columns contains -1'
		df_not_reply['inReplyToStatusId']=np.nan

		df=pd.concat([df_reply,df_not_reply],ignore_index=True)
		df['inReplyToStatusId']=df['inReplyToStatusId'].astype('str')
		df['retweetedStatus.id']=df['retweetedStatus.id'].astype('str')
		df['date']=df['createdAt'].dt.date


		df['text'] = df['text'].str.replace('[^A-Za-z]', ' ', flags=re.UNICODE)
		df_tweet_count=pd.DataFrame([[tag,yesterday,df.shape[0]]],columns=["tag","date","tweet_count"])

		df['tweet_date']=df['createdAt'].dt.date

		df_retweets=df[~df['retweetedStatus.user.screenName'].isnull()]
		df_reply=df[~df['inReplyToScreenName'].isnull()]
		# Creating Dataframe
		d1={'metric':['total_tweets','retweets','replies','direct_tweets'],'count':[df.shape[0],df_retweets.shape[0],df_reply.shape[0],abs(df.shape[0]-(df_retweets.shape[0]+df_reply.shape[0]))]}
		df_tweet_dist=pd.DataFrame(d1)
		df_tweet_dist['date']=yesterday
		df_tweet_dist['tag']=tag

		stop_words = set(stopwords.words("english"))
		new_words = ['amp','co', 'http','youtube','https','www','com','href','result ','search','query','result', 'br', 'channel', 'audience','hai', 'sir', 'youtuber','video','film','please','language','ki','ka','p','se','movie','mera','help','le','de','la','que','un','en','di','il','si','el','che','et','da','pa','na','lo','mi','tu','ko','ne','ya','con','te','je','pc','e','n','l','j','f','k','c','r','du','b','eu','al','qui','sa','ha','rt']
		stop_words = stop_words.union(new_words)
		corp = []
		for i in range(0, len(df)):
			#Remove punctuations
			#text = re.sub('[^a-zA-Z]', ' ', df['text'][i])
			text = re.sub('[^a-zA-Z]', ' ', str(df['text'].iloc[i]))
			text=text.lower()
			##Convert to list from string
			text = text.split()
			##Lemmatizing
			lm = WordNetLemmatizer() 
			text = [lm.lemmatize(word) for word in text if not word in stop_words] 
			text = " ".join(text)
			corp.append(text)

		#Removing stopwords from the text column
		df['text']=df['text'].astype('str')
		df["text"]=df["text"].apply(lambda words: ' '.join(word.lower() for word in words.split() if word not in stop_words))

		cv=CountVectorizer(max_df=0.7, stop_words=stop_words, ngram_range=(1,2), min_df=0.001)
		X=cv.fit_transform(corp)
		vector = cv.transform(corp)

		#Most frequently occuring words
		def get_top_n_words(corpus, n=None):
			vec = cv.fit(corp)
			bag_of_words = vec.transform(corp)
			sum_words = bag_of_words.sum(axis=0) 
			words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
			words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
			return words_freq[:n]

		top_words = get_top_n_words(corp, n=20)
		top_df = pd.DataFrame(top_words)
		top_df.columns=["word", "freq"]
		top_df['date']=yesterday
		top_df=top_df[["date","word","freq"]]
		drop_words=['please','others','ponniyin','selvan','ponniyin selvan']
		top_df=top_df[~top_df['word'].isin(drop_words)]
		top_df=top_df.head(5)
		top_df['tag']=tag

		return df_tweet_count, df_tweet_dist,top_df

	def pushdata(df_tweet_count, df_tweet_dist,top_df):

			engine = create_engine('postgresql://psone_flash_base:psone@kalacitra.in/psone_flash_base')
			df_tweet_count.to_sql('tw_tweetcount',engine,if_exists='append',index=False, method="multi")
			df_tweet_dist.to_sql('tw_tweetdist',engine,if_exists='append',index=False, method="multi")
			top_df.to_sql('tw_freq_words',engine,if_exists='append',index=False, method="multi")
			engine.dispose()

	df_dumped = fetch(uri,mydb)
	df_tweet_count, df_tweet_dist,top_df = transform(df_dumped)
	pushdata(df_tweet_count, df_tweet_dist,top_df)
    
uri = sys.argv[1]
mydb = sys.argv[2]
tags = sys.argv[3]
tags = tags.split(',')
for tag in tags:
	twitter_competitor_analysis_weekly(uri,mydb,tag)
