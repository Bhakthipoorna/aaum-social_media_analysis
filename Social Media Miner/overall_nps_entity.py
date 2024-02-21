def overall_nps_entity():

	import pandas as pd
	import pymongo as pym
	import psycopg2
	from sqlalchemy import create_engine
	from datetime import datetime,timedelta
	import pytz
	IST = pytz.timezone('Asia/Kolkata')
	Today=pd.to_datetime(datetime.strftime(datetime.now(IST), '%Y-%m-%d'))
	Today=datetime.strftime(Today, '%Y-%m-%d')
	yesterday=datetime.strftime(datetime.now(IST) - timedelta(1), '%Y-%m-%d')


	def fetch():

		conn = psycopg2.connect(database="psone_flash_base", user='psone_flash_base', password='psone', host='kalacitra.in')  
		cursor = conn.cursor()
		tw_nps=pd.read_sql("select * from tw_nps where date between '{}' and '{}' and tag!='Naane Varuvean' and tag!='Vikram Vedha' ".format(yesterday,Today),conn)
		tw_entity=pd.read_sql("select * from tw_entity where date between '{}' and '{}' and tag!='Naane Varuvean' and tag!='Vikram Vedha' ".format(yesterday,Today),conn)
		yt_nps=pd.read_sql("select * from yt_nps where date between '{}' and '{}' ".format(yesterday,Today),conn)
		yt_entity=pd.read_sql("select * from yt_entity where tag!='Competitor-NaaneVaruvean' and  date between '{}' and '{}'  ".format(yesterday,Today),conn)
		yt_video_counts=pd.read_sql("select * from yt_views_overtime where date between '{}' and '{}'".format(yesterday,Today),conn)
		cursor.close()
		conn.close()

		return tw_nps, tw_entity, yt_nps, yt_entity, yt_video_counts

	def transform(tw_nps, tw_entity, yt_nps, yt_entity, yt_video_counts):

		tw_nps=(pd.DataFrame(tw_nps.groupby('date')['nps'].mean())).reset_index()
		tw_nps['tag']='Overall'
		tw_entity=(pd.DataFrame(tw_entity.groupby(['date','entity'])['percentage'].mean())).reset_index()
		tw_entity['tag']='Overall'
		yt_nps=(pd.DataFrame(yt_nps.groupby('date')['nps'].mean())).reset_index()
		yt_nps['tag']='Overall'
		yt_entity=(pd.DataFrame(yt_entity.groupby(['date','entity'])['percentage'].mean())).reset_index()
		yt_entity['tag']='Overall'


		yt_video_counts=pd.DataFrame(yt_video_counts[['tag','language']].value_counts()).reset_index()
		yt_video_counts=yt_video_counts[['tag','language']]
		yt_video_counts['count']=1
		yt_video_counts=yt_video_counts.sort_values('tag')
		yt_video_counts['date']=yesterday

		return tw_nps, tw_entity, yt_nps, yt_entity,yt_video_counts

	def pushdata(tw_nps, tw_entity, yt_nps, yt_entity, yt_video_counts):

		engine = create_engine('postgresql://psone_flash_base:psone@kalacitra.in/psone_flash_base')
		tw_nps.to_sql('tw_nps',engine,if_exists='append',index=False, method="multi")
		tw_entity.to_sql('tw_entity',engine,if_exists='append',index=False, method="multi")
		yt_nps.to_sql('yt_nps',engine,if_exists='append',index=False, method="multi")
		yt_entity.to_sql('yt_entity',engine,if_exists='append',index=False, method="multi")
		yt_video_counts.to_sql('yt_video_count',engine,if_exists='append',index=False, method="multi")
		engine.dispose()

	tw_nps, tw_entity, yt_nps, yt_entity, yt_video_counts = fetch()
	tw_nps, tw_entity, yt_nps, yt_entity, yt_video_counts = transform(tw_nps, tw_entity, yt_nps, yt_entity, yt_video_counts)
	pushdata(tw_nps, tw_entity, yt_nps, yt_entity, yt_video_counts)


