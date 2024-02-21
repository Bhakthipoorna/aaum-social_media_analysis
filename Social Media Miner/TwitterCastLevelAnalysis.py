def TwitterCastLevelAnalysis():

	import pandas as pd
	import numpy as np
	import datetime
	import psycopg2
	from sqlalchemy import create_engine
	from dateutil.relativedelta import relativedelta
	my_date = datetime.date.today() # if date is 01/01/2018
	year, week, day_of_week = my_date.isocalendar()
	last_date_of_week = pd.to_datetime(datetime.date(year, 1, 1) + relativedelta(weeks=+week))
	last_date_of_week=str(last_date_of_week)
	last_date_of_week=last_date_of_week[0:10]

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

		df_cast=pd.concat([df_karthi,df_yayamravi,df_trisha,df_arrahman,df_vikramprabhu,df_ravivarman,df_jeyamnohan,df_lyca_production,df_madras_talkies],ignore_index=True)
		df_cast['text']=df_cast['text'].str.lower()
		df_castwise_tweets=pd.DataFrame(df_cast['screen_name'].value_counts()).reset_index()
		df_castwise_tweets.columns=['screen_name','total_tweets']
		df_cast=df_cast[df_cast['text'].str.contains('ps1|ps 1|ps-1|ponniyinselvan|ponniyin selvan|ponni nadhi|chola|ponninadhi|ps1trailer|ps1audio')]
		df_cast1=df_cast[['id_str','text','screen_name','name','created_at','retweeted','retweet_count','followers_count','favorite_count']]

		df_report=df_cast1.groupby(['screen_name','name']).agg({'id_str':'count','retweet_count':sum,'followers_count':max,'favorite_count':sum})
		df_report=df_report.reset_index()
		df_report.rename(columns={'id_str':'RelatedTweets'},inplace=True)
		df_report['AvgRetweet/tweet']=round(df_report['retweet_count']/df_report['RelatedTweets'])

		df_report=df_report[["screen_name","name","RelatedTweets","retweet_count","followers_count","favorite_count","AvgRetweet/tweet"]]
		df_report.rename(columns={"RelatedTweets":"related_tweets","AvgRetweet/tweet":"avg_retweet_per_tweet"},inplace=True)
		cast_modified_names=['Maniratnam','Vikram','Jayam Ravi','Karthi','Aishwarya Rai Bachchan','Trisha Krishnan','Vikram Prabhu','Ravi Varman','A Sreekar Prasad','A R Rahman','B Jeyamohan','Lyca Productions','Madras Talkies']
		df_report['name'].replace({'maniratnam.official':'Maniratnam','Jayam Ravi':'Jayam Ravi','Actor Karthi':'Karthi','aishwaryaraibachchan_arb':'Aishwarya Rai Bachchan','Trish':'Trisha Krishnan','A.R.Rahman':'A R Rahman','Ravi varman':'Ravi Varman','Vikram Prabhu':'Vikram Prabhu',},inplace=True)
		df_report.rename(columns={'name':'cast_name'},inplace=True)
		cast_modified_names=pd.DataFrame(cast_modified_names)
		cast_modified_names.columns=['cast_name']
		df_castwise_summary=pd.merge(cast_modified_names,df_report,on='cast_name',how='left')
		df_castwise_summary.fillna(0,inplace=True)

		return df_castwise_summary

	def pushdata(df_castwise_summary):

		conn = psycopg2.connect(database="psone_flash_base", user='psone', password='psone', host='localhost')  
		cursor = conn.cursor()
		delete1="DELETE FROM df_castwise_summary"
		cursor.execute(delete1)
		conn.commit()

		engine = create_engine('postgresql://psone:psone@localhost/psone_flash_base')
		df_castwise_summary.to_sql('df_castwise_summary',engine,if_exists='append',index=False, method="multi")
		engine.dispose()

	df_castwise_summary  = fetch_and_transform()
	#pushdata(df_castwise_summary)

	return df_castwise_summary
df_castwise_summary = TwitterCastLevelAnalysis()