import sys
def twitter_analysis_weekly(uri,mydb,tag):

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

		if tag =='Movie':
			hashtag_tweets= mydb['hashtag_tweets_one']#hashtag_tweets#hashtag_tweets_songs
		elif tag =='Trailer':
			hashtag_tweets= mydb['hashtag_tweets_one']#hashtag_tweets#hashtag_tweets_songs
		elif tag =='Ponni':
			hashtag_tweets= mydb['hashtag_tweets_songs']#hashtag_tweets#hashtag_tweets_songs
		elif tag =='Chola':
			hashtag_tweets= mydb['hashtag_tweets_songs']#hashtag_tweets#hashtag_tweets_songs
		elif tag =='Trisha':
			hashtag_tweets= mydb['hashtag_tweets_one']#hashtag_tweets#hashtag_tweets_songs
		elif tag =='Karthi':
			hashtag_tweets= mydb['hashtag_tweets_one']#hashtag_tweets#hashtag_tweets_songs
		elif tag =='Vikram':
			hashtag_tweets= mydb['hashtag_tweets_one']#hashtag_tweets#hashtag_tweets_songs
		elif tag =='Aishwarya Rai':
			hashtag_tweets= mydb['hashtag_tweets_one']#hashtag_tweets#hashtag_tweets_songs
		elif tag =='Jayam Ravi':
			hashtag_tweets= mydb['hashtag_tweets_one']#hashtag_tweets#hashtag_tweets_songs
		elif tag =='Maniratnam':
			hashtag_tweets= mydb['hashtag_tweets_one']#hashtag_tweets#hashtag_tweets_songs
		elif tag =='AR Rahman':
			hashtag_tweets= mydb['hashtag_tweets_one']#hashtag_tweets#hashtag_tweets_songs
		elif tag =='Review':
			hashtag_tweets= mydb['hashtag_tweets_one']#hashtag_tweets#hashtag_tweets_songs
		
		

		config_file='tw_config.ini'
		config = ConfigParser()
		config.read(config_file)

		if tag =='Movie':
			key_words=config['PS1']['teaser_keywords']
		elif tag =='Trailer':
			key_words=config['TRAILER']['Trailer_keywords']
		elif tag =='Ponni':
			key_words=config['SONG_PONNINADHI']['songs_ponninadhi_keywords']
		elif tag =='Chola':
			key_words=config['SONG_CHOLACHOLA']['songs_cholachola_keywords']
		elif tag =='Trisha':
			key_words=config['TRISHA']['trisha_keywords']
		elif tag =='Karthi':
			key_words=config['KARTHI']['karthi_keywords']
		elif tag =='Vikram':
			key_words=config['VIKRAM']['vikram_keywords']
		elif tag =='Aishwarya Rai':
			key_words=config['AISHWARYA_RAI']['aishwaryarai_keywords']
		elif tag =='Jayam Ravi':
			key_words=config['JAYAM_RAVI']['jayamravi_keywords']
		elif tag =='Maniratnam':
			key_words=config['MANIRATNAM']['maniratnam_keywords']
		elif tag =='AR Rahman':
			key_words=config['ARRAHMAN']['arrahman_keywords']
		elif tag =='Review':
			key_words=config['REVIEW']['review_keywords']
		
		
		df_dumped= pd.json_normalize(hashtag_tweets.find())

		return df_dumped, key_words

	def transform(df_dumped, key_words):

		df_dumped[["createdAt"]]=df_dumped[["createdAt"]].apply(pd.to_datetime, errors='coerce') 
		df_dumped=df_dumped.sort_values('createdAt')

		#Dropping the tweets with empty Id and empty CreatedAt
		df_dumped=df_dumped[~df_dumped['createdAt'].isnull()]
		df_dumped=df_dumped[df_dumped['createdAt']>=prior_7days]
		df_dumped=df_dumped[df_dumped['createdAt']<Today]
		df_dumped=df_dumped[~df_dumped['id'].isnull()]

		df_dumped['id']=df_dumped['id'].astype('str')
		#Dropping duplicates
		df=df_dumped.drop_duplicates(['id'],keep='last')

		df['text']=df['text'].str.lower()
		df['text'] = df['text'].str.replace('[^A-Za-z0-9]', ' ', flags=re.UNICODE)
		df=df[df['text'].str.contains(key_words)]
		df['text'] = df['text'].str.replace('[^A-Za-z]', ' ', flags=re.UNICODE)

		if len(df)<1:

			df_tweet_count=pd.DataFrame([[yesterday,0,tag]],columns=['date','tweet_count','tag'])
			df_tweet_dist=pd.DataFrame(['total_tweets','replies','direct_tweets','retweets'])
			df_tweet_dist['tag']=tag
			df_tweet_dist['date']=yesterday
			df_tweet_dist['count']=0
			top_df=pd.DataFrame(columns=['date','tag','word','freq'])
			df_popular=pd.DataFrame(columns=['date','tag','influencer_screenname','retweets_count','user_followers_count'])
			df_more_retweet=pd.DataFrame(columns=['date','tag','user_screenname','reach','fan_base'])
			df_positive_profile=pd.DataFrame(columns=['date','tag','user_name','nps','tweet_count','followers_count'])
			df_negative_profile=pd.DataFrame(columns=['date','tag','user_name','nps','tweet_count','followers_count'])

		else:


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

			df['favoriteCount']=pd.to_numeric(df['favoriteCount'], errors='coerce').convert_dtypes()
			df['retweetCount']=pd.to_numeric(df['retweetCount'], errors='coerce').convert_dtypes()
			df['user.followersCount']=pd.to_numeric(df['user.followersCount'], errors='coerce').convert_dtypes()
			df['user.friendsCount']=pd.to_numeric(df['user.friendsCount'], errors='coerce').convert_dtypes()
			df['user.favouritesCount']=pd.to_numeric(df['user.favouritesCount'], errors='coerce').convert_dtypes()

			df['date']=df['createdAt'].dt.date

			df_tweet_count=pd.DataFrame([[yesterday,df.shape[0]]],columns=["date","tweet_count"])
			df_tweet_count['tag']=tag

			df['tweet_date']=df['createdAt'].dt.date

			df_retweets=df[~df['retweetedStatus.user.screenName'].isnull()]
			df_reply=df[~df['inReplyToScreenName'].isnull()]
			# Creating Dataframe
			d1={'metric':['total_tweets','retweets','replies','direct_tweets'],'count':[df.shape[0],df_retweets.shape[0],df_reply.shape[0],df.shape[0]-(df_retweets.shape[0]+df_reply.shape[0])]}
			df_tweet_dist=pd.DataFrame(d1)
			df_tweet_dist['date']=yesterday
			df_tweet_dist['tag']=tag
			df_tweet_dist=df_tweet_dist[["date","tag","metric","count"]]
			df_favourite_count=pd.DataFrame(df.groupby('date')['user.favouritesCount'].sum()).reset_index()

			df_followers_count=pd.DataFrame(df.groupby(['user.screenName'])['user.followersCount'].max()).reset_index()

			df_top_influencer=(pd.DataFrame(df['retweetedStatus.user.screenName'].value_counts().head(15))).reset_index()
			df_top_influencer.rename(columns={"retweetedStatus.user.screenName":"retweets_count","index":"user.screenName"},inplace=True)
			df_popular=pd.merge(df_top_influencer,df_followers_count,on='user.screenName',how='left')
			df_popular.rename(columns={'user.screenName':'influencer_screenname','user.followersCount':'user_followers_count'},inplace=True)
			df_popular['date']=yesterday
			df_popular=df_popular[['date','influencer_screenname','retweets_count','user_followers_count']]
			df_popular['tag']=tag

			df_more_retweets=(pd.DataFrame(df_retweets['user.screenName'].value_counts())).reset_index()
			df_more_retweets.columns=['user.screenName','retweeted_count']
			df_more_retweets=pd.merge(df_more_retweets,df_followers_count,on='user.screenName',how='left')
			df_more_retweets['date']=yesterday
			df_more_retweets=df_more_retweets[['date','user.screenName','retweeted_count','user.followersCount']]
			df_more_retweet=df_more_retweets.sort_values('retweeted_count',ascending=False).head(15)
			df_more_retweet.rename(columns={"user.screenName":"user_screenname","retweeted_count":"reach","user.followersCount":"fan_base"},inplace=True)
			df_more_retweet['tag']=tag

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
			top_df['tag']=tag
			top_df=top_df.head(5)

			df['text']=df['text'].astype('str')
			df['senti']=df['text'].apply(sid.polarity_scores)

			df['compsenti'] = df['senti'].apply(lambda x: x['compound'])
			df['possenti'] = df['senti'].apply(lambda x: x['pos'])
			df['negsenti'] = df['senti'].apply(lambda x: x['neg'])
			df['nps'] = df['possenti']-df['negsenti']

			df_nps=pd.DataFrame(df.groupby('date')['nps'].mean()).reset_index()

			df_profile_wise=pd.DataFrame(df.groupby('user.screenName')['nps'].mean()).reset_index()
			df_profile_wise=pd.merge(df_profile_wise,df_more_retweets,on='user.screenName')
			df_profile_wise=df_profile_wise[df_profile_wise['nps']>0]
			df_positive_profile=df_profile_wise.sort_values(['retweeted_count','nps'],ascending=False).head(10)
			df_positive_profile['tag']=tag
			df_positive_profile.rename(columns={'retweeted_count':'tweet_count','user.screenName':'user_name','user.followersCount':'followers_count'},inplace=True)
			df_negative_profile=df_profile_wise.sort_values('retweeted_count',ascending=False)
			df_negative_profile=df_negative_profile[df_negative_profile['nps']<0]
			df_negative_profile=df_negative_profile.head(10)
			df_negative_profile['tag']=tag
			df_negative_profile.rename(columns={'retweeted_count':'tweet_count','user.screenName':'user_name','user.followersCount':'followers_count'},inplace=True)
	 
		return df_tweet_count, df_tweet_dist,top_df, df_popular, df_more_retweet,df_positive_profile, df_negative_profile

	def pushdata(df_tweet_count, df_tweet_dist,top_df, df_popular, df_more_retweet,df_positive_profile, df_negative_profile):

			engine = create_engine('postgresql://psone_flash_base:psone@kalacitra.in/psone_flash_base')
			df_tweet_count.to_sql('tw_tweetcount',engine,if_exists='append',index=False, method="multi")
			df_tweet_dist.to_sql('tw_tweetdist',engine,if_exists='append',index=False, method="multi")
			top_df.to_sql('tw_freq_words',engine,if_exists='append',index=False, method="multi")
			df_popular.to_sql('tw_popular',engine,if_exists='append',index=False, method="multi")
			df_more_retweet.to_sql('tw_more_tweets',engine,if_exists='append',index=False, method="multi")
			df_positive_profile.to_sql('tw_positive_profile',engine,if_exists='append',index=False, method="multi")
			df_negative_profile.to_sql('tw_negative_profile',engine,if_exists='append',index=False, method="multi")
			engine.dispose()

	df_dumped, key_words = fetch(uri,mydb)
	df_tweet_count, df_tweet_dist,top_df, df_popular, df_more_retweet,df_positive_profile, df_negative_profile = transform(df_dumped, key_words)
	pushdata(df_tweet_count, df_tweet_dist,top_df, df_popular, df_more_retweet,df_positive_profile, df_negative_profile)
    
uri = sys.argv[1]
mydb = sys.argv[2]
tags = sys.argv[3]
tags = tags.split(',')
for tag in tags:
	twitter_analysis_weekly(uri,mydb,tag)
