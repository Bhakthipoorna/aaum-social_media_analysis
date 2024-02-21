
def celebrity_tracker():
	
	import pandas as pd
	import numpy as np
	import psycopg2
	from sqlalchemy import create_engine
	from math import log, floor

	def fetch():

		conn = psycopg2.connect(database="psone_flash_base", user='psone_flash_base', password='psone', host='kalacitra.in')  
		cursor = conn.cursor()
		df_insta=pd.read_sql("select * from insta_cast_analysis",conn)
		df_twitter=pd.read_sql("select * from tw_cast_analysis",conn)

		return df_insta, df_twitter, conn, cursor

	def transform(df_insta, df_twitter, conn, cursor):

		df_insta.rename(columns={'followers_count':'insta_followers_count'},inplace=True)
		df_insta.fillna(0,inplace=True)
		df_twitter.rename(columns={'followers_count':'twitter_followers_count'},inplace=True)
		df_twitter.fillna(0,inplace=True)

		df=pd.merge(df_insta,df_twitter, on='cast_name')
		df['followers']=df['insta_followers_count']+df['twitter_followers_count']
		df['twitter_reach']=(((df['total_tweets']+df['favorite_count'])/df['related_tweets'])/df['twitter_followers_count'])*100
		df['instagram_reach']=(((df['likes']+df['views']+df['comments'])/df['related_posts'])/df['insta_followers_count'])*100
		df=df[['cast_name','followers','related_tweets','related_posts','twitter_reach','instagram_reach']]
		df.rename(columns={'related_tweets':'tweets','related_posts':'posts'},inplace=True)
		df.fillna(0,inplace=True)
		df[['followers','tweets','posts']]=df[['followers','tweets','posts']].astype('int')
		df[['twitter_reach','instagram_reach']]=df[['twitter_reach','instagram_reach']].round(2)
		df.rename(columns={'cast_name':'celebrity_name','tweets':'tweet_post','posts':'insta_post','twitter_reach':'twitter_reach','instagram_reach':'instagram_reach'},inplace=True)
		df=df.sort_values('followers',ascending=False)
		df_morethan_k=df[df['followers']>+1000]
		df_lessthan_k=df[df['followers']<1000]


		def human_format(number):
			units = ['', 'K', 'M', 'G', 'T', 'P']
			k = 1000.0
			magnitude = int(floor(log(number, k)))
			return '%.2f%s' % (number / k**magnitude, units[magnitude])

		df_morethan_k['followers']=df_morethan_k['followers'].apply(human_format)
		df=pd.concat([df_morethan_k,df_lessthan_k],ignore_index=False)
		df
		return df

	def pushdata(df):

		delete1="DELETE FROM celebrity_track"
		cursor.execute(delete1)
		conn.commit()
		cursor.close()
		conn.close()

		engine = create_engine('postgresql://psone_flash_base:psone@kalacitra.in/psone_flash_base')
		df.to_sql('celebrity_track',engine,if_exists='append',index=False, method="multi")
		engine.dispose()

	df_insta, df_twitter, conn, cursor = fetch()
	df = transform(df_insta, df_twitter, conn, cursor)
	pushdata(df)
