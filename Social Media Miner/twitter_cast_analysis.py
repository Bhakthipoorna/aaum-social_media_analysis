def twitter_cast_analysis():

	import pandas as pd
	import numpy as np
	import datetime
	import psycopg2
	from sqlalchemy import create_engine

	def fetch_and_transform():

		df_karthi=pd.read_csv('Karthi_All_Tweets.csv')
		df_yayamravi=pd.read_csv('JayamRavi_All_Tweets.csv')
		df_trisha=pd.read_csv('Trisha_All_Tweets.csv')
		df_arrahman=pd.read_csv('ARRahman_All_Tweets.csv')
		df_vikramprabhu=pd.read_csv('VikramPrabhu_All_Tweets.csv')
		df_ravivarman=pd.read_csv('RaviVarman.csv')
		df_jeyamnohan=pd.read_csv('JeyaMohan.csv')
		df_lyca_production=pd.read_csv('LycaProduction_All_Tweets.csv')
		df_madras_talkies=pd.read_csv('MadrasTalkies_All_Tweets.csv')
		df_prakashraj=pd.read_csv('PrakashRaj_All_Tweets.csv')


		df_cast=pd.concat([df_karthi,df_yayamravi,df_trisha,df_arrahman,df_vikramprabhu,df_ravivarman,df_jeyamnohan,df_lyca_production,df_madras_talkies,df_prakashraj],ignore_index=True)
		df_cast['text']=df_cast['text'].str.lower()
		df_castwise_tweets=pd.DataFrame(df_cast['screen_name'].value_counts()).reset_index()
		df_castwise_tweets.columns=['cast_name','total_tweets']
		df_cast=df_cast[df_cast['text'].str.contains('ps1|ps 1|ps-1|ponniyinselvan|ponniyin selvan|ponni nadhi|chola|ponninadhi|ps1trailer|ps1audio')]
		df_cast1=df_cast[['id_str','text','screen_name','name','created_at','retweeted','retweet_count','followers_count','favorite_count']]

		df_report=df_cast1.groupby(['screen_name','name']).agg({'id_str':'count','retweet_count':sum,'followers_count':max,'favorite_count':sum})
		df_report=df_report.reset_index()
		df_report.rename(columns={'id_str':'RelatedTweets'},inplace=True)
		df_report['AvgRetweet/tweet']=round(df_report['retweet_count']/df_report['RelatedTweets'])

		df_report=df_report[["screen_name","name","RelatedTweets","retweet_count","followers_count","favorite_count","AvgRetweet/tweet"]]
		df_report.rename(columns={"RelatedTweets":"related_tweets","AvgRetweet/tweet":"avg_retweet_per_tweet"},inplace=True)
		cast_modified_names=['Maniratnam','Vikram','Jayam Ravi','Karthi','Aishwarya Rai Bachchan','Trisha Krishnan','Vikram Prabhu','Ravi Varman','A Sreekar Prasad','A R Rahman','B Jeyamohan','Lyca Productions','Madras Talkies','Prakash Raj']
		df_report['screen_name'].replace({'maniratnam.official':'Maniratnam','actor_jayamravi':'Jayam Ravi','Karthi_Offl':'Karthi','aishwaryaraibachchan_arb':'Aishwarya Rai Bachchan','trishtrashers':'Trisha Krishnan','arrahman':'A R Rahman','dop_ravivarman':'Ravi Varman','iamVikramPrabhu':'Vikram Prabhu','MadrasTalkies_':'Madras Talkies','LycaProductions':'Lyca Productions','prakashraaj':'Prakash Raj'},inplace=True)
		df_report.rename(columns={'screen_name':'cast_name'},inplace=True)
		cast_modified_names=pd.DataFrame(cast_modified_names)
		cast_modified_names.columns=['cast_name']
		df_castwise_summary=pd.merge(cast_modified_names,df_report,on='cast_name',how='left')
		df_castwise_tweets['cast_name'].replace({'maniratnam.official':'Maniratnam','actor_jayamravi':'Jayam Ravi','Karthi_Offl':'Karthi','aishwaryaraibachchan_arb':'Aishwarya Rai Bachchan','trishtrashers':'Trisha Krishnan','arrahman':'A R Rahman','dop_ravivarman':'Ravi Varman','iamVikramPrabhu':'Vikram Prabhu','MadrasTalkies_':'Madras Talkies','LycaProductions':'Lyca Productions','prakashraaj':'Prakash Raj'},inplace=True)
		df_castwise_summary=pd.merge(df_castwise_summary,df_castwise_tweets,on='cast_name',how='left')
		df_castwise_summary=df_castwise_summary[~df_castwise_summary['cast_name'].isnull()]
		df_castwise_summary.fillna(0,inplace=True)
		df_castwise_summary=df_castwise_summary[['cast_name','total_tweets','related_tweets','retweet_count','followers_count','favorite_count','avg_retweet_per_tweet']]
		df_cast_tweets=df_cast[['screen_name','text']]
		df_cast_tweets.rename(columns={'screen_name':'name'},inplace=True)
		df_cast_tweets['name'].replace({'maniratnam.official':'Maniratnam','actor_jayamravi':'Jayam Ravi','Karthi_Offl':'Karthi','aishwaryaraibachchan_arb':'Aishwarya Rai Bachchan','trishtrashers':'Trisha Krishnan','arrahman':'A R Rahman','dop_ravivarman':'Ravi Varman','iamVikramPrabhu':'Vikram Prabhu','MadrasTalkies_':'Madras Talkies','LycaProductions':'Lyca Productions','prakashraaj':'Prakash Raj'},inplace=True)
		df_cast_tweets.rename(columns={'text':'tweet'},inplace=True)

		return df_castwise_summary, df_cast_tweets

	def pushdata(df_castwise_summary,df_cast_tweets):

		conn = psycopg2.connect(database="psone_flash_base", user='psone_flash_base', password='psone', host='kalacitra.in')  
		cursor = conn.cursor()
		delete1="DELETE FROM tw_cast_analysis"
		cursor.execute(delete1)
		conn.commit()
		delete2="DELETE FROM tw_tweets"
		cursor.execute(delete2)
		conn.commit()
		cursor.close()
		conn.close()

		engine = create_engine('postgresql://psone_flash_base:psone@kalacitra.in/psone_flash_base')
		df_castwise_summary.to_sql('tw_cast_analysis',engine,if_exists='append',index=False, method="multi")
		df_cast_tweets.to_sql('tw_tweets',engine,if_exists='append',index=False, method="multi")
		engine.dispose()

	df_castwise_summary, df_cast_tweets  = fetch_and_transform()
	pushdata(df_castwise_summary, df_cast_tweets)
