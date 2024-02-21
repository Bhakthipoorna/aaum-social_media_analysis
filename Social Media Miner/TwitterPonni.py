def TwitterPonni(uri,mydb):

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
	my_date = datetime.date.today() # if date is 01/01/2018
	year, week, day_of_week = my_date.isocalendar()
	last_date_of_week = pd.to_datetime(datetime.date(year, 1, 1) + relativedelta(weeks=+week))
	last_date_of_week=str(last_date_of_week)
	last_date_of_week=last_date_of_week[0:10]
	previous_week = pd.to_datetime(datetime.date(year, 1, 1) + relativedelta(weeks=week-2))
	previous_week=str(previous_week)
	previous_week=previous_week[0:10]

	def fetch_and_transform(uri,mydb):

		uri = uri #"mongodb://localhost:27017"
		client= pym.MongoClient(uri)
		mydb = client[mydb] #client['socialmedia']
		hashtag_tweets= mydb['hashtag_tweets_songs']#hashtag_tweets#hashtag_tweets_songs
		df_dumped= pd.json_normalize(hashtag_tweets.find())
		#df_dumped=pd.read_csv('ps_new.csv')

		config_file='tw_config.ini'
		config = ConfigParser()
		config.read(config_file)
		key_words=config['SONG_PONNINADHI']['songs_ponninadhi_keywords']

		df_dumped[["createdAt"]]=df_dumped[["createdAt"]].apply(pd.to_datetime, errors='coerce') 
		df_dumped=df_dumped.sort_values('createdAt')

		#Dropping the tweets with empty Id and empty CreatedAt
		df_dumped=df_dumped[~df_dumped['createdAt'].isnull()]
		df_dumped=df_dumped[df_dumped['createdAt']>previous_week]
		df_dumped=df_dumped[df_dumped['createdAt']<last_date_of_week]
		df_dumped=df_dumped[~df_dumped['id'].isnull()]

		df_dumped['id']=df_dumped['id'].astype('str')
		#Dropping duplicates
		df=df_dumped.drop_duplicates(['id'],keep='last')

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

		df['text']=df['text'].str.lower()
		df['text'] = df['text'].str.replace('[^A-Za-z0-9]', ' ', flags=re.UNICODE)
		df=df[df['text'].str.contains(key_words)]
		df['text'] = df['text'].str.replace('[^A-Za-z]', ' ', flags=re.UNICODE)
		df_tweet_count=pd.DataFrame([[last_date_of_week,df.shape[0]]],columns=["date","tweet_count"])
		df_tweet_count['tag']=tag

		df['tweet_date']=df['createdAt'].dt.date

		df_retweets=df[~df['retweetedStatus.user.screenName'].isnull()]
		df_reply=df[~df['inReplyToScreenName'].isnull()]
		# Creating Dataframe
		df_tweet_dist = pd.DataFrame([[df.shape[0],df_retweets.shape[0],df_reply.shape[0],(df.shape[0]-(df_retweets.shape[0]-df_reply.shape[0]))]],columns =["total_tweets","retweets","replies","direct_tweets"])
		df_tweet_dist['date']=last_date_of_week
		df_tweet_dist=df_tweet_dist[["date","total_tweets","retweets","replies","direct_tweets"]]

		df_daily_tweet_count=pd.DataFrame(df['tweet_date'].value_counts()).reset_index()
		df_daily_tweet_count.columns=['date','total_tweets']
		df_daily_tweet_count=df_daily_tweet_count.sort_values('date')
		df_daily_tweet_count['date']=df_daily_tweet_count['date'].astype('object')

		df_daily_retweet=pd.DataFrame(df_retweets.groupby('tweet_date')['createdAt'].count()).reset_index()
		df_daily_retweet.columns=['date','retweets']
		df_daily_retweet['date']=df_daily_retweet['date'].astype('object')
		df_daily_replies=pd.DataFrame(df_reply.groupby('tweet_date')['createdAt'].count()).reset_index()
		df_daily_replies.columns=['date','replies']
		df_daily_replies['date']=df_daily_retweet['date'].astype('object')

		df_retweets_reply=pd.merge(df_daily_retweet,df_daily_replies,on='date',how='outer')
		df_daily_tweetdist=pd.merge(df_retweets_reply,df_daily_tweet_count,on='date',how='outer')
		df_daily_tweetdist.fillna(0,inplace=True)
		df_daily_tweetdist['direct_tweets']=df_daily_tweetdist['total_tweets']-(df_daily_tweetdist['retweets']+df_daily_tweetdist['replies'])
		df_daily_tweet_count.rename(columns={'total_tweets':'tweets_count'},inplace=True)
		df_daily_tweet_count['tag']=tag

		df_direct_tweets=df_daily_tweetdist[['date','direct_tweets']]
		df_favourite_count=pd.DataFrame(df.groupby('date')['user.favouritesCount'].sum()).reset_index()
	
		df_reach=(pd.merge(df_direct_tweets,df_favourite_count,on='date',how='outer')).merge(df_daily_retweet,on='date',how='outer')
		df_reach=df_reach[df_reach['direct_tweets']>0]
		df_reach['reach']=(df_reach['retweets']+df_reach['user.favouritesCount'])
		df_reach['reach']=df_reach['reach']/df_reach['direct_tweets']
		df_reach['reach'] = df_reach['reach'].astype(int, errors='ignore')
		df_reach['date']=pd.to_datetime(df_reach['date'])
		df_reach=df_reach.fillna(0)
		df_reach['tag']=tag
		df_reach=df_reach[['date','tag','reach']]
		

		df_followers_count=pd.DataFrame(df.groupby(['user.screenName'])['user.followersCount'].max()).reset_index()

		df_top_influencer=(pd.DataFrame(df['retweetedStatus.user.screenName'].value_counts().head(15))).reset_index()
		df_top_influencer.rename(columns={"retweetedStatus.user.screenName":"retweets_count","index":"user.screenName"},inplace=True)
		df_popular=pd.merge(df_top_influencer,df_followers_count,on='user.screenName',how='left')
		df_popular.rename(columns={'user.screenName':'influencer_screenname','user.followersCount':'user_followers_count'},inplace=True)
		df_popular['date']=last_date_of_week
		df_popular=df_popular[['date','influencer_screenname','retweets_count','user_followers_count']]

		df_more_retweets=(pd.DataFrame(df_retweets['user.screenName'].value_counts())).reset_index()
		df_more_retweets.columns=['user.screenName','retweeted_count']
		df_more_retweets=pd.merge(df_more_retweets,df_followers_count,on='user.screenName',how='left')
		df_more_retweets['date']=last_date_of_week
		df_more_retweets=df_more_retweets[['date','user.screenName','retweeted_count','user.followersCount']]
		df_more_retweet=df_more_retweets.sort_values('retweeted_count',ascending=False).head(15)
		df_more_retweet.rename(columns={"user.screenName":"user_screenname","user.followersCount":"user_followers_count"},inplace=True)

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
		top_df['date']=last_date_of_week
		top_df=top_df[["date","word","freq"]]
		drop_words=['please','others','ponniyin','selvan','ponniyin selvan']
		top_df=top_df[~top_df['word'].isin(drop_words)]
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
		df_positive_profile=df_profile_wise.sort_values(['retweeted_count','nps'],ascending=False).head(10)
		df_positive_profile['tag']=tag
		df_positive_profile.rename(columns={'retweeted_count':'tweet_count','user.screenName':'user_name','user.followersCount':'followers_count'},inplace=True)
		df_negative_profile=df_profile_wise.sort_values('retweeted_count',ascending=False)
		df_negative_profile=df_negative_profile[df_negative_profile['nps']<0]
		df_negative_profile=df_negative_profile.head(10)
		df_negative_profile['tag']=tag
		df_negative_profile.rename(columns={'retweeted_count':'tweet_count','user.screenName':'user_name','user.followersCount':'followers_count'},inplace=True)
		

		df=df.set_index('createdAt') #df_more_retweets

		df['vfx'] = df['text'].str.contains('vfx',case = False)
		vfx=pd.DataFrame(df['vfx'].resample('1d').sum().reset_index())
		vfx.rename(columns={'createdAt':'date'},inplace=True)
		vfx['date']=vfx['date'].dt.date
		vfx['date']=pd.to_datetime(vfx['date'])
		vfx_daywise=pd.DataFrame(df.groupby(by='date').vfx.count()).reset_index()
		vfx_daywise['date']=pd.to_datetime(vfx_daywise['date'])
		merged_vfx= pd.merge(vfx, vfx_daywise, on='date', how= 'outer' )
		merged_vfx['att_sc']= merged_vfx['vfx_x']/merged_vfx['vfx_y']
		merged_vfx.fillna(0,inplace=True)
		merged_vfx=merged_vfx[['date','att_sc']]
		merged_vfx['entity']='vfx'

		df['grand'] = df['text'].str.contains('grand',case = False)
		grand=pd.DataFrame(df['grand'].resample('1d').sum().reset_index())
		grand.rename(columns={'createdAt':'date'},inplace=True)
		grand['date']=grand['date'].dt.date
		grand['date']=pd.to_datetime(grand['date'])
		grand_daywise=pd.DataFrame(df.groupby(by='date').grand.count()).reset_index()
		grand_daywise['date']=pd.to_datetime(grand_daywise['date'])
		merged_grand= pd.merge(grand, grand_daywise, on='date', how= 'outer' )
		merged_grand['att_sc']= merged_grand['grand_x']/merged_grand['grand_y']
		merged_grand.fillna(0,inplace=True)
		merged_grand=merged_grand[['date','att_sc']]
		merged_grand['entity']='grand'

		df['director'] = df['text'].str.contains('director',case = False)
		director=pd.DataFrame(df['director'].resample('1d').sum().reset_index())
		director.rename(columns={'createdAt':'date'},inplace=True)
		director['date']=director['date'].dt.date
		director['date']=pd.to_datetime(director['date'])
		director_daywise=pd.DataFrame(df.groupby(by='date').director.count()).reset_index()
		director_daywise['date']=pd.to_datetime(director_daywise['date'])
		merged_director= pd.merge(director, director_daywise, on='date', how= 'outer' )
		merged_director['att_sc']= merged_director['director_x']/merged_director['director_y']
		merged_director.fillna(0,inplace=True)
		merged_director=merged_director[['date','att_sc']]
		merged_director['entity']='director'


		df['music'] = df['text'].str.contains('music',case = False)
		music=pd.DataFrame(df['music'].resample('1d').sum().reset_index())
		music.rename(columns={'createdAt':'date'},inplace=True)
		music['date']=music['date'].dt.date
		music['date']=pd.to_datetime(music['date'])
		music_daywise=pd.DataFrame(df.groupby(by='date').music.count()).reset_index()
		music_daywise['date']=pd.to_datetime(music_daywise['date'])
		merged_music= pd.merge(music, music_daywise, on='date', how= 'outer' )
		merged_music['att_sc']= merged_music['music_x']/merged_music['music_y']
		merged_music.fillna(0,inplace=True)
		merged_music=merged_music[['date','att_sc']]
		merged_music['entity']='music'


		df['camera'] = df['text'].str.contains('camera',case = False)
		camera=pd.DataFrame(df['camera'].resample('1d').sum().reset_index())
		camera.rename(columns={'createdAt':'date'},inplace=True)
		camera['date']=camera['date'].dt.date
		camera['date']=pd.to_datetime(camera['date'])
		camera_daywise=pd.DataFrame(df.groupby(by='date').camera.count()).reset_index()
		camera_daywise['date']=pd.to_datetime(camera_daywise['date'])
		merged_camera= pd.merge(camera, camera_daywise, on='date', how= 'outer' )
		merged_camera['att_sc']= merged_camera['camera_x']/merged_camera['camera_y']
		merged_camera.fillna(0,inplace=True)
		merged_camera=merged_camera[['date','att_sc']]
		merged_camera['entity']='camera'


		df['sets'] = df['text'].str.contains('sets',case = False)
		sets=pd.DataFrame(df['sets'].resample('1d').sum().reset_index())
		sets.rename(columns={'createdAt':'date'},inplace=True)
		sets['date']=sets['date'].dt.date
		sets['date']=pd.to_datetime(sets['date'])
		sets_daywise=pd.DataFrame(df.groupby(by='date').sets.count()).reset_index()
		sets_daywise['date']=pd.to_datetime(sets_daywise['date'])
		merged_sets= pd.merge(sets, sets_daywise, on='date', how= 'outer' )
		merged_sets['att_sc']= merged_sets['sets_x']/merged_sets['sets_y']
		merged_sets.fillna(0,inplace=True)
		merged_sets=merged_sets[['date','att_sc']]
		merged_sets['entity']='sets'


		df['history'] = df['text'].str.contains('history',case = False)
		history=pd.DataFrame(df['history'].resample('1d').sum().reset_index())
		history.rename(columns={'createdAt':'date'},inplace=True)
		history['date']=history['date'].dt.date
		history['date']=pd.to_datetime(history['date'])
		history_daywise=pd.DataFrame(df.groupby(by='date').history.count()).reset_index()
		history_daywise['date']=pd.to_datetime(history_daywise['date'])
		merged_history= pd.merge(history, history_daywise, on='date', how= 'outer' )
		merged_history['att_sc']= merged_history['history_x']/merged_history['history_y']
		merged_history.fillna(0,inplace=True)
		merged_history=merged_history[['date','att_sc']]
		merged_history['entity']='history'

		df_entity=pd.concat([merged_vfx,merged_grand,merged_director,merged_music,merged_camera,merged_sets,merged_history],ignore_index=False)
		df_entity.rename(columns={'att_sc':'score'},inplace=True)
		df_entity['tag']=tag
		df_entity=df_entity.sort_values('date')

		return df_tweet_count, df_daily_tweet_count, df_tweet_dist,df_daily_tweetdist, top_df, df_nps, df_reach, df_popular, df_more_retweet, df_entity,df_positive_profile, df_negative_profile

	def pushdata(df_tweet_count, df_tweet_dist, top_df, df_nps, df_reach, df_popular, df_more_retweet, df_entity):

			engine = create_engine('postgresql://psone:psone@localhost/psone_flash_base')
			df_tweet_count.to_sql('tw_tweetcount_history',engine,if_exists='append',index=False, method="multi")
			df_tweet_dist.to_sql('tw_tweetdist_history',engine,if_exists='append',index=False, method="multi")
			top_df.to_sql('tw_freq_words_history',engine,if_exists='append',index=False, method="multi")
			df_nps['attime']=(datetime.datetime.now(IST)).strftime("%Y-%m-%d")
			df_nps.to_sql('tw_nps_history',engine,if_exists='append',index=False, method="multi")
			df_reach.to_sql('tw_reach_history',engine,if_exists='append',index=False, method="multi")
			df_popular.to_sql('tw_popular_history',engine,if_exists='append',index=False, method="multi")
			df_more_retweet.to_sql('tw_more_tweets_history',engine,if_exists='append',index=False, method="multi")
			df_entity.to_sql('tw_entity_history',engine,if_exists='append',index=False, method="multi")
			engine.dispose()

	df_tweet_count, df_daily_tweet_count, df_tweet_dist,df_daily_tweetdist, top_df, df_nps, df_reach, df_popular, df_more_retweet, df_entity, df_positive_profile, df_negative_profile= fetch_and_transform(uri,mydb)
	#pushdata(df_tweet_count, df_tweet_dist, top_df, df_nps, df_reach, df_popular, df_more_retweet, df_entity)

	return df_tweet_count, df_daily_tweet_count, df_tweet_dist,df_daily_tweetdist, top_df, df_nps, df_reach, df_popular, df_more_retweet, df_entity, df_positive_profile, df_negative_profile
df_tweet_count, df_daily_tweet_count, df_tweet_dist,df_daily_tweetdist, top_df, df_nps, df_reach, df_popular, df_more_retweet, df_entity, df_positive_profile, df_negative_profile  = TwitterPonni(uri,mydb)