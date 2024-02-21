def twitter_media_handles_analysis(uri,mydb):
	import pandas as pd
	import numpy as np
	import re
	from datetime import datetime
	import pymongo as pym
	from math import log, floor
	import psycopg2
	from sqlalchemy import create_engine

	def fetch(uri, mydb):

		uri = uri
		client= pym.MongoClient(uri)
		mydb = client[mydb]
		media_tweets= mydb['page_tweets_media']
		df_dumped= pd.json_normalize(media_tweets.find())
		client.close()

		return df_dumped

	def transform(df_dumped):

		df=df_dumped.drop_duplicates('id',keep='last')
		df['text']=df['text'].str.lower()
		df=df[df['text'].str.contains('ps1|ps 1|ps-1|ponniyinselvan|ponniyin|selvan|ponni|ponninadhi|chola')]
		
		df=df[['id','text','user.screenName','user.name','createdAt','isRetweeted','retweetCount','user.followersCount','user.favouritesCount']]
		df_report=df.groupby(['user.screenName','user.name']).agg({'id':'count','retweetCount':sum,'user.followersCount':max,'user.favouritesCount':sum})
		df_report=df_report.reset_index()
		df_report.rename(columns={'id':'RelatedTweets'},inplace=True)
		df_report['twitter_reach']=(((df_report['retweetCount']+df_report['user.favouritesCount'])/df_report['RelatedTweets'])/df_report['user.followersCount'])*100
		df_report.rename(columns={'user.name':'name','RelatedTweets':'tweets','retweetCount':'retweet_count','user.followersCount':'followers_count','user.favouritesCount':'favourites_count'},inplace=True)
		df_report=df_report[['name','tweets','retweet_count','followers_count','favourites_count','twitter_reach']]

		df_morethan_k=df_report[df_report['followers_count']>+1000]
		df_lessthan_k=df_report[df_report['followers_count']<1000]


		def human_format(number):
		    units = ['', 'K', 'M', 'G', 'T', 'P']
		    k = 1000.0
		    magnitude = int(floor(log(number, k)))
		    return '%.2f%s' % (number / k**magnitude, units[magnitude])

		df_morethan_k['followers_count']=df_morethan_k['followers_count'].apply(human_format)
		df_report=pd.concat([df_morethan_k,df_lessthan_k],ignore_index=False)
		df_tweets=df[['user.name','text']]
		df_tweets.rename(columns={'user.name':'name','text':'tweet'},inplace=True)

		return df_report, df_tweets

	def pushdata(df_report, df_tweets):

		conn = psycopg2.connect(database="psone_flash_base", user='psone_flash_base', password='psone', host='kalacitra.in')  
		cursor = conn.cursor()
		delete1="DELETE FROM tw_media_handles"
		cursor.execute(delete1)
		conn.commit()
		cursor.close()
		conn.close()

		engine = create_engine('postgresql://psone_flash_base:psone@kalacitra.in/psone_flash_base')
		df_report.to_sql('tw_media_handles',engine,if_exists='append',index=False, method="multi")
		df_tweets.to_sql('tw_tweets',engine,if_exists='append',index=False, method="multi")
		engine.dispose()

	df_dumped = fetch(uri, mydb)
	df_report, df_tweets = transform(df_dumped)
	pushdata(df_report, df_tweets)
