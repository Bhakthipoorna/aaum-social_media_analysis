def milestone_analysis(uri,mydb):
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
	Today=datetime.strftime(Today, '%Y-%m-%d')
	yesterday=datetime.strftime(datetime.now(IST) - timedelta(1), '%Y-%m-%d')

	def fetch(uri, mydb):     
	    
		uri = uri#'mongodb://localhost:27017'
		client= pym.MongoClient(uri)
		mydb = client[mydb] #client['socialmedia']
		#hashtag_tweets= mydb['hashtag_tweets']#hashtag_tweets#hashtag_tweets_songs
		hashtag_tweets_songs= mydb['hashtag_tweets_songs']#hashtag_tweets#hashtag_tweets_songs
		#df_dumped1= pd.json_normalize(hashtag_tweets.find())
		df_dumped_songs= pd.json_normalize(hashtag_tweets_songs.find())
		client.close()

		#df_dumped=pd.concat([df_dumped1, df_dumped_songs],ignore_index=False)
		df_dumped=df_dumped_songs

		config_file='tw_config.ini'
		config = ConfigParser()
		config.read(config_file)
		key_words=config['ALL']['all_keywords']

		return df_dumped, key_words
	df_dumped, key_words = fetch(uri, mydb)


	def transform(df_dumped, key_words):

		df_dumped[["createdAt"]]=df_dumped[["createdAt"]].apply(pd.to_datetime, errors='coerce') 
		df_dumped=df_dumped.sort_values('createdAt')

		df_dumped=df_dumped[~df_dumped['createdAt'].isnull()]
		df_dumped=df_dumped[~df_dumped['id'].isnull()]

		df_dumped=df_dumped[~df_dumped['createdAt'].isnull()]
		df_dumped=df_dumped[df_dumped['createdAt']>yesterday]
		df_dumped=df_dumped[df_dumped['createdAt']<Today]
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
		df['tweet_date']=df['createdAt'].dt.date


		df['text']=df['text'].astype('str')
		df['senti']=df['text'].apply(sid.polarity_scores)

		df['compsenti'] = df['senti'].apply(lambda x: x['compound'])
		df['possenti'] = df['senti'].apply(lambda x: x['pos'])
		df['negsenti'] = df['senti'].apply(lambda x: x['neg'])
		df['nps'] = df['possenti']-df['negsenti']

		df=df.set_index('createdAt') #df_more_retweets

		df['vfx'] = df['text'].str.contains('vfx',case = False)
		vfx=df['vfx'].resample('24d').sum()
		df_vfx=pd.DataFrame([['vfx', sum(vfx)*100/ df.shape[0]]],columns=['Entity','Score']).reset_index()

		df['grand'] = df['text'].str.contains('grand',case = False)
		grand= df['grand'].resample('24d').sum()
		df_grand=pd.DataFrame([['grand', sum(grand)*100/ df.shape[0]]],columns=['Entity','Score']).reset_index()

		df['director'] = df['text'].str.contains('director',case = False)
		director= df['director'].resample('24d').sum()
		df_director=pd.DataFrame([['director', sum(director)*100/ df.shape[0]]],columns=['Entity','Score']).reset_index()

		df['music'] = df['text'].str.contains('music|arrahman|rahman|arr|a r rahman',case = False)
		music= df['music'].resample('24d').sum()
		df_music=pd.DataFrame([['music', sum(music)*100/ df.shape[0]]],columns=['Entity','Score']).reset_index()

		df['camera'] = df['text'].str.contains('camera',case = False)
		camera=df['camera'].resample('24d').sum()
		df_camera=pd.DataFrame([['camera', sum(camera)*100/ df.shape[0]]],columns=['Entity','Score']).reset_index()

		df['sets'] = df['text'].str.contains('set',case = False)
		sets= df['sets'].resample('24d').sum()
		df_sets=pd.DataFrame([['sets', sum(sets)*100/ df.shape[0]]],columns=['Entity','Score']).reset_index()

		df['history'] = df['text'].str.contains('history',case = False)
		history = df['history'].resample('24d').sum()
		df_history=pd.DataFrame([['history', sum(history)*100/ df.shape[0]]],columns=['Entity','Score']).reset_index()

		#df1=df.reset_index()

		df['date']=pd.to_datetime(df['date'])

		df_vfx=df[df['vfx']==True]
		df_grand=df[df['grand']==True]
		df_director=df[df['director']==True]
		df_music=df[df['music']==True]
		df_camera=df[df['camera']==True]
		df_sets=df[df['sets']==True]
		df_history=df[df['history']==True]

		df_vfx=pd.DataFrame(df_vfx.groupby('date')['nps'].mean()).reset_index()
		df_vfx['attribute']='vfx'
		df_grand=pd.DataFrame(df_grand.groupby('date')['nps'].mean()).reset_index()
		df_grand['attribute']='grand'
		df_director=pd.DataFrame(df_director.groupby('date')['nps'].mean()).reset_index()
		df_director['attribute']='director'
		df_music=pd.DataFrame(df_music.groupby('date')['nps'].mean()).reset_index()
		df_music['attribute']='music'
		df_camera=pd.DataFrame(df_camera.groupby('date')['nps'].mean()).reset_index()
		df_camera['attribute']='camera'
		df_sets=pd.DataFrame(df_sets.groupby('date')['nps'].mean()).reset_index()
		df_sets['attribute']='sets'
		df_history=pd.DataFrame(df_history.groupby('date')['nps'].mean()).reset_index()
		df_history['attribute']='history'

		dates = pd.date_range(start=yesterday, end=Today)
		df_date = pd.DataFrame(dates,columns=['date'])
		cond = [df_date['date']<='2022-07-28',(df_date['date']>'2022-07-28')&(df_date['date']<='2022-08-16'),(df_date['date']>'2022-08-16')&(df_date['date']<='2022-09-03'),df_date['date']>'2022-09-03']
		choice = ['Teaser','PonniNadhiSong','CholaCholaSong','Trailer']
		df_date['milestone'] = np.select(cond,choice)

		df_vfx=pd.merge(df_vfx,df_date['date'],on='date',how='outer')
		df_vfx['attribute'].fillna('vfx',inplace=True)
		df_grand=pd.merge(df_grand,df_date['date'],on='date',how='outer')
		df_grand['attribute'].fillna('grand',inplace=True)
		df_director=pd.merge(df_director,df_date['date'],on='date',how='outer')
		df_director['attribute'].fillna('director',inplace=True)
		df_music=pd.merge(df_music,df_date['date'],on='date',how='outer')
		df_music['attribute'].fillna('music',inplace=True)
		df_camera=pd.merge(df_camera,df_date['date'],on='date',how='outer')
		df_camera['attribute'].fillna('camera',inplace=True)
		df_sets=pd.merge(df_sets,df_date['date'],on='date',how='outer')
		df_sets['attribute'].fillna('sets',inplace=True)
		df_history=pd.merge(df_history,df_date['date'],on='date',how='outer')
		df_history['attribute'].fillna('history',inplace=True)

		df_attributes=pd.concat([df_vfx,df_grand,df_director,df_music,df_camera,df_sets,df_history],ignore_index=False)
		df_attributes['date']=pd.to_datetime(df_attributes['date'])
		df_milestone=pd.merge(df_date,df_attributes,on='date')
		df_milestone['nps'].fillna(0,inplace=True)
		df_milestone=df_milestone[['date','milestone','attribute','nps']]
		df_milestone=df_milestone[df_milestone['date']<Today]

		return df_milestone

	def pushdata(df_milestone):

		engine = create_engine('postgresql://psone_flash_base:psone@kalacitra.in/psone_flash_base')
		df_milestone.to_sql('tw_milestone',engine,if_exists='append',index=False, method="multi")
		engine.dispose()

	df_milestone = transform(df_dumped, key_words)
	pushdata(df_milestone)
    
	return df_milestone
df_milestone = milestone_analysis(uri,mydb)