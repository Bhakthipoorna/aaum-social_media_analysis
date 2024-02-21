import sys
def instagram_analysis(uri,mydb):
	import pandas as pd
	import numpy as np
	import nltk
	import os
	import json
	import re
	import pymongo as pym
	from nltk.sentiment.vader import SentimentIntensityAnalyzer
	from nltk.corpus import stopwords
	from nltk.stem.wordnet import WordNetLemmatizer
	sid = SentimentIntensityAnalyzer()
	import psycopg2
	from sqlalchemy import create_engine
	from configparser import ConfigParser
	from datetime import datetime,timedelta
	import pytz
	IST = pytz.timezone('Asia/Kolkata')
	Today=pd.to_datetime(datetime.strftime(datetime.now(IST), '%Y-%m-%d'))
	yesterday=datetime.strftime(datetime.now(IST) - timedelta(1), '%Y-%m-%d')
	prev_4days=datetime.strftime(datetime.now(IST) - timedelta(5), '%Y-%m-%d')

	def fetch(uri,mydb):
		uri =uri#"mongodb://localhost:27017"
		client= pym.MongoClient(uri)
		mydb = client[mydb]
		insta_comments= mydb['Manual_Data_Push_For_Instagram_Comment_Scapper']
		df_insta= pd.json_normalize(insta_comments.find())
		client.close()

		return df_insta

	def transform(df_insta):

		df_lyca_production_lookup=pd.read_excel("Insta_PS1_lookup_lyca_productions.xlsx",engine='openpyxl')
		df_madras_talkies_lookup=pd.read_excel("Insta_PS1_lookup_madras_talkies.xlsx",engine='openpyxl')
		df_jayamravi_lookup=pd.read_excel("Insta_PS1_lookup_jayamravi.xlsx",engine='openpyxl')
		df_aishwaryarai_lookup=pd.read_excel("Insta_PS1_lookup_aishwaryarai.xlsx",engine='openpyxl')
		df_trisha_lookup=pd.read_excel("Insta_PS1_lookup_trisha.xlsx",engine='openpyxl')
		df_karthi_lookup=pd.read_excel("Insta_PS1_lookup_karthi.xlsx",engine='openpyxl')
		df_arrahman_lookup=pd.read_excel("Insta_PS1_lookup_arrahman.xlsx",engine='openpyxl')
		df_manirathnam_lookup=pd.read_excel("Insta_PS1_lookup_maniratnam.xlsx",engine='openpyxl')
		df_prakashraj_lookup=pd.read_excel('Insta_PS1_lookup_prakashraj.xlsx',engine='openpyxl')

		df_lookup=pd.concat([df_lyca_production_lookup,df_madras_talkies_lookup,df_jayamravi_lookup,df_aishwaryarai_lookup,df_trisha_lookup,df_karthi_lookup,df_arrahman_lookup,df_manirathnam_lookup, df_prakashraj_lookup],ignore_index=False)

		df_lookup['Image_Url']=df_lookup['Image_Url'].str.strip()
		df_lookup['Image_Url']=df_lookup['Image_Url'].str.lstrip()
		df_lookup['Image_Url']=df_lookup['Image_Url'].str.rstrip()
		df_lookup.drop_duplicates('Image_Url',keep='last',inplace=True)
		df_insta['Image_Url']=df_insta['Image_Url'].str.strip()
		df_insta['Image_Url']=df_insta['Image_Url'].str.lstrip()
		df_insta['Image_Url']=df_insta['Image_Url'].str.rstrip()

		df_lookup_image=df_lookup[~df_lookup['Image_Url'].str.contains('reel/',na=False)]

		df_lookup_video=df_lookup[df_lookup['Image_Url'].str.contains('reel/',na=False)] # In the lookup page video links contains 'reels/' in the link but the in the collection video links does not contain 'reel/'
		df_lookup_video['Image_Url']=df_lookup_video['Image_Url'].str.replace("reel/","p/") #Ex: 'https://www.instagram.com/p/CfgEMTlDMTo/' in the collection and 'https://www.instagram.com/reels/CfgEMTlDMTo/' in the lookup page

		df_lookup_image['Image_Url']=df_lookup_image['Image_Url'].str[0:-28] # Url in lookup page conatins '?utm_source=ig_web_copy_link' at the end, we are removing this.
		df_lookup_video['Image_Url']=df_lookup_video['Image_Url'].str[0:-28]

		df_picture=df_insta[df_insta['Image_Url'].isin(df_lookup_image['Image_Url'])]
		df_video=df_insta[df_insta['Image_Url'].isin(df_lookup_video['Image_Url'])]
		df_picture=pd.merge(df_picture,df_lookup_image,on='Image_Url',how='left')
		df_video=pd.merge(df_video,df_lookup_video,on='Image_Url',how='left')
		df_picture=df_picture[df_picture['Date of Post']<str(Today)]
		df_video=df_video[df_video['Date of Post']<str(Today)]
		df_picture=df_picture[df_picture['Date of Post']>=prev_4days]
		df_video=df_video[df_video['Date of Post']>=prev_4days]


		df_picture["Likes Counts"]=df_picture["Likes Counts"].str.lower()
		df_picture['Likes Counts']=df_picture['Likes Counts'].str.strip()
		df_picture['Likes Counts']=df_picture['Likes Counts'].str.lstrip()
		df_picture['Likes Counts']=df_picture['Likes Counts'].str.rstrip()
		df_picture['Likes Counts']=df_picture['Likes Counts'].str[-18:]

		df_picture['Likes Counts']=df_picture['Likes Counts'].str.replace('[^0-9]',' ', flags=re.UNICODE) #(K for thousand and M for million)
		df_video['Likes Counts']=df_video['Likes Counts'].str.replace('[^0-9]',' ', flags=re.UNICODE) #(K for thousand and M for million)
		df_picture['Likes Counts']=df_picture['Likes Counts'].str.replace(" ","")
		df_picture['Likes']=df_picture['Likes Counts']

		df_video['Likes Counts']=df_video['Likes Counts'].str.strip()
		df_video['Likes Counts']=df_video['Likes Counts'].str.lstrip()
		df_video['Likes Counts']=df_video['Likes Counts'].str.rstrip()
		df_video['Likes Counts']=df_video['Likes Counts'].str.replace(" ","")
		df_video['Likes Counts']=df_video['Likes Counts'].str[-18:]
		df_video['Views']=df_video['Likes Counts']

		df=pd.concat([df_picture,df_video],ignore_index=False)

		df['Date of Post']=pd.to_datetime(df['Date of Post'])
		df['Likes']=df['Likes'].fillna(0)
		df['Views']=df['Views'].fillna(0)
		df['Comments Counts']=df['Comments Counts'].fillna(0)

		df['date']=df['Date of Post'].dt.date
		
		df_post_count=pd.DataFrame(df.groupby('date')['Image_Url'].nunique()).reset_index()
		df_post_count.columns=['date','posts_count']
		df_post_count

		#summary table by post
		df_summary=pd.DataFrame(df.groupby(['Image_Url','url_description'])['Likes','Views','Comments Counts'].max()).reset_index()
		df_summary.rename(columns={'Comments Counts':'Comments'},inplace=True)
		df_summary['date']=yesterday
		df_summary=df_summary[['date','Image_Url',"url_description",'Likes','Views','Comments']]
		df_summary.rename(columns={"Image_Url":"url","Likes":"likes","Views":"views","Comments":"comments"},inplace=True)
		df_summary['likes']=pd.to_numeric(df_summary['likes'], errors='coerce').convert_dtypes()
		df_summary['views']=pd.to_numeric(df_summary['views'], errors='coerce').convert_dtypes()
		df_summary['comments']=pd.to_numeric(df_summary['comments'], errors='coerce').convert_dtypes()
		df_summary.rename(columns={'url':'description_url','url_description':'description'},inplace=True)
		df_summary=df_summary.head(5)

		total_likes=df_summary['likes'].sum()
		total_views=df_summary['views'].sum()
		total_comments=df_summary['comments'].sum()
		#df_reach=df_reach[df_reach['date']<='2022-09-11']

		df_reach=pd.DataFrame(df.groupby(['date','Image_Url'])['Likes','Views','Comments Counts'].max()).reset_index()
		df_reach['Likes']=pd.to_numeric(df_reach['Likes'], errors='coerce').convert_dtypes()
		df_reach['Views']=pd.to_numeric(df_reach['Views'], errors='coerce').convert_dtypes()
		df_reach['Comments Counts']=pd.to_numeric(df_reach['Comments Counts'], errors='coerce').convert_dtypes()
		df_reach=pd.DataFrame(df_reach.groupby('date')['Likes','Views','Comments Counts'].sum()).reset_index()
		df_reach['reach']=df_reach['Likes']+df_reach['Views']+df_reach['Comments Counts']
		df_reach=df_reach[['date','reach']]
		
		stop_words = set(stopwords.words("english"))
		new_words = ['amp','co', 'http','youtube','http','www','com','href','result ','search','query','result', 'br', 'channel', 'audience','hai', 'sir', 'youtuber']
		stop_words = stop_words.union(new_words)

		corp = []
		for i in range(0, df.shape[0]):
		    #Remove punctuations
		    text = re.sub('[^a-zA-Z]', ' ', str(df['Comments'].iloc[i])) 
		    text=text.lower()
		    ##Convert to list from string
		    text = text.split()
		    ##Lemmatizing
		    lm = WordNetLemmatizer() 
		    text = [lm.lemmatize(word) for word in text if not word in stop_words] 
		    text = " ".join(text)
		    corp.append(text) 
		df['clean_comments'] = np.array(corp)
		df['Comments']=df['Comments'].astype('str')
		df["Comments"]=df["Comments"].apply(lambda words: ' '.join(word.lower() for word in words.split() if word not in stop_words))
		df['Comments']=df['Comments'].str.replace('[^A-Za-z]', ' ', flags=re.UNICODE, regex= True)
		df['Comments']=df['Comments'].str.lower()


		from sklearn.feature_extraction.text import CountVectorizer
		cv=CountVectorizer(max_df=0.7,
		                   stop_words=stop_words, 
		                   ngram_range=(1,2), 
		                   min_df=0.001)
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
		top_df=top_df[['date','word','freq']]
		drop_words=['please','others','ponniyin','selvan','ponniyin selvan','lyca','production','madras','talkies']
		top_df=top_df[~top_df['word'].isin(drop_words)]
		top_df=top_df.head(5)

		df['clean_comments']=df['clean_comments'].astype('str')
		df['senti']=df['clean_comments'].apply(sid.polarity_scores)

		df['compsenti'] = df['senti'].apply(lambda x: x['compound'])
		df['possenti'] = df['senti'].apply(lambda x: x['pos'])
		df['negsenti'] = df['senti'].apply(lambda x: x['neg'])
		df['nps']=df['possenti']-df['negsenti']
		df['date']=df['Date of Post'].dt.date
		df_nps_over_time=pd.DataFrame(df.groupby('date')['nps','possenti','negsenti'].mean()).reset_index()


		nps=df_nps_over_time[['date','nps']]
		nps['metric']='nps'
		nps.rename(columns={'nps':'score'},inplace=True)
		possenti=df_nps_over_time[['date','possenti']]
		possenti['metric']='possenti'
		possenti.rename(columns={'possenti':'score'},inplace=True)
		negsenti=df_nps_over_time[['date','negsenti']]
		negsenti['metric']='negsenti'
		negsenti.rename(columns={'negsenti':'score'},inplace=True)

		df_sentiment=pd.concat([nps,possenti,negsenti],ignore_index=False)
		df_sentiment=df_sentiment.fillna(0)
		df_sentiment=df_sentiment[['date','metric','score']]
		df_sentiment['metric'].replace({'possenti':'positive','negsenti':'negative'},inplace=True)

		#most positive posts
		df_positive_post=(pd.DataFrame(df.groupby(by= ['Image_Url','url_description'])[ 'possenti'].mean().sort_values(ascending = False).head(5))).reset_index()
		df_positive_post.rename(columns={"Image_Url":"url"},inplace=True)
		df_positive_post['date']=yesterday
		df_positive_post=df_positive_post[["date","url","url_description","possenti"]]
		df_positive_post.rename(columns={'url':'description_url','url_description':'description','possenti':'positive_sentiment'},inplace=True)


		#most negative posts
		df_negative_post=(pd.DataFrame(df.groupby(by= ['Image_Url','url_description'])[ 'negsenti'].mean().sort_values(ascending = False).head())).reset_index()
		df_negative_post.rename(columns={"Image_Url":"url"},inplace=True)
		df_negative_post['date']=yesterday
		df_negative_post=df_negative_post[["date","url","url_description","negsenti"]]
		df_negative_post.rename(columns={'url':'description_url','url_description':'description','negsenti':'negative_sentiment'},inplace=True)

		#most positive profiles by NPS
		df_nps_per_person=(pd.DataFrame(df.groupby(by= 'Commented_Person_name')[ 'nps'].mean())).reset_index()
		df_nps_per_person.columns=["person_name","nps"] 
		df_nps_per_person['date']=yesterday
		df_nps_per_person=df_nps_per_person[["date","person_name","nps"]]
		df_nps_per_person=df_nps_per_person.sort_values('nps',ascending=False)

		#most active profiles 
		df_most_active_person=pd.DataFrame(df.Commented_Person_name.value_counts().sort_values(ascending=False))
		df_most_active_person=df_most_active_person.reset_index()
		df_most_active_person.rename(columns={'Commented_Person_name':'No. Of comments from the person','index':'person_name'},inplace=True)
		df_most_active_person

		#Most active profile with nps
		df_most_active_perfile_and_positive_nps=(pd.merge(df_most_active_person,df_nps_per_person,on='person_name',how='outer')).sort_values('No. Of comments from the person',ascending=False)
		df_most_active_perfile_and_positive_nps=df_most_active_perfile_and_positive_nps[df_most_active_perfile_and_positive_nps['nps']>0]
		df_most_active_perfile_and_positive_nps=df_most_active_perfile_and_positive_nps.head(10)
		df_most_active_perfile_and_positive_nps.rename(columns={'No. Of comments from the person':'comments_count','person_name':'profile_name'},inplace=True)
		df_most_active_perfile_and_negative_nps=(pd.merge(df_most_active_person,df_nps_per_person,on='person_name',how='outer')).sort_values(['No. Of comments from the person','nps'],ascending=[False,False])
		df_most_active_perfile_and_negative_nps=df_most_active_perfile_and_negative_nps[df_most_active_perfile_and_negative_nps['nps']<0]
		df_most_active_perfile_and_negative_nps=df_most_active_perfile_and_negative_nps.head(10)
		df_most_active_perfile_and_negative_nps.rename(columns={'No. Of comments from the person':'comments_count','person_name':'profile_name'},inplace=True)
		df_most_active_perfile_and_negative_nps #(insta_positive_profile, insta_negative_profile) - table names.


		df=df.set_index('Date of Post')

		df['vfx'] = df['Comments'].str.contains('vfx',case = False)
		vfx=pd.DataFrame(df['vfx'].resample('1d').sum().reset_index())
		vfx.rename(columns={'Date of Post':'date'},inplace=True)
		vfx['date']=vfx['date'].dt.date
		vfx['date']=pd.to_datetime(vfx['date'])
		vfx_daywise=pd.DataFrame(df.groupby(by='date').vfx.count()).reset_index()
		vfx_daywise['date']=pd.to_datetime(vfx_daywise['date'])
		merged_vfx= pd.merge(vfx, vfx_daywise, on='date', how= 'outer' )
		merged_vfx['att_sc']= merged_vfx['vfx_x']/merged_vfx['vfx_y']
		merged_vfx.fillna(0,inplace=True)
		merged_vfx=merged_vfx[['date','att_sc']]
		merged_vfx['entity']='vfx'

		df['grand'] = df['Comments'].str.contains('grand',case = False)
		grand=pd.DataFrame(df['grand'].resample('1d').sum().reset_index())
		grand.rename(columns={'Date of Post':'date'},inplace=True)
		grand['date']=grand['date'].dt.date
		grand['date']=pd.to_datetime(grand['date'])
		grand_daywise=pd.DataFrame(df.groupby(by='date').grand.count()).reset_index()
		grand_daywise['date']=pd.to_datetime(grand_daywise['date'])
		merged_grand= pd.merge(grand, grand_daywise, on='date', how= 'outer' )
		merged_grand['att_sc']= merged_grand['grand_x']/merged_grand['grand_y']
		merged_grand.fillna(0,inplace=True)
		merged_grand=merged_grand[['date','att_sc']]
		merged_grand['entity']='grand'

		df['director'] = df['Comments'].str.contains('director',case = False)
		director=pd.DataFrame(df['director'].resample('1d').sum().reset_index())
		director.rename(columns={'Date of Post':'date'},inplace=True)
		director['date']=director['date'].dt.date
		director['date']=pd.to_datetime(director['date'])
		director_daywise=pd.DataFrame(df.groupby(by='date').director.count()).reset_index()
		director_daywise['date']=pd.to_datetime(director_daywise['date'])
		merged_director= pd.merge(director, director_daywise, on='date', how= 'outer' )
		merged_director['att_sc']= merged_director['director_x']/merged_director['director_y']
		merged_director.fillna(0,inplace=True)
		merged_director=merged_director[['date','att_sc']]
		merged_director['entity']='director'


		df['music'] = df['Comments'].str.contains('music',case = False)
		music=pd.DataFrame(df['music'].resample('1d').sum().reset_index())
		music.rename(columns={'Date of Post':'date'},inplace=True)
		music['date']=music['date'].dt.date
		music['date']=pd.to_datetime(music['date'])
		music_daywise=pd.DataFrame(df.groupby(by='date').music.count()).reset_index()
		music_daywise['date']=pd.to_datetime(music_daywise['date'])
		merged_music= pd.merge(music, music_daywise, on='date', how= 'outer' )
		merged_music['att_sc']= merged_music['music_x']/merged_music['music_y']
		merged_music.fillna(0,inplace=True)
		merged_music=merged_music[['date','att_sc']]
		merged_music['entity']='music'


		df['camera'] = df['Comments'].str.contains('camera',case = False)
		camera=pd.DataFrame(df['camera'].resample('1d').sum().reset_index())
		camera.rename(columns={'Date of Post':'date'},inplace=True)
		camera['date']=camera['date'].dt.date
		camera['date']=pd.to_datetime(camera['date'])
		camera_daywise=pd.DataFrame(df.groupby(by='date').camera.count()).reset_index()
		camera_daywise['date']=pd.to_datetime(camera_daywise['date'])
		merged_camera= pd.merge(camera, camera_daywise, on='date', how= 'outer' )
		merged_camera['att_sc']= merged_camera['camera_x']/merged_camera['camera_y']
		merged_camera.fillna(0,inplace=True)
		merged_camera=merged_camera[['date','att_sc']]
		merged_camera['entity']='camera'


		df['sets'] = df['Comments'].str.contains('sets',case = False)
		sets=pd.DataFrame(df['sets'].resample('1d').sum().reset_index())
		sets.rename(columns={'Date of Post':'date'},inplace=True)
		sets['date']=sets['date'].dt.date
		sets['date']=pd.to_datetime(sets['date'])
		sets_daywise=pd.DataFrame(df.groupby(by='date').sets.count()).reset_index()
		sets_daywise['date']=pd.to_datetime(sets_daywise['date'])
		merged_sets= pd.merge(sets, sets_daywise, on='date', how= 'outer' )
		merged_sets['att_sc']= merged_sets['sets_x']/merged_sets['sets_y']
		merged_sets.fillna(0,inplace=True)
		merged_sets=merged_sets[['date','att_sc']]
		merged_sets['entity']='sets'


		df['history'] = df['Comments'].str.contains('history',case = False)
		history=pd.DataFrame(df['history'].resample('1d').sum().reset_index())
		history.rename(columns={'Date of Post':'date'},inplace=True)
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
		df_entity.rename(columns={'att_sc':'percentage'},inplace=True)
		df_entity['percentage'] = df_entity['percentage']*100
		df_entity=df_entity.sort_values('date')

		return df_post_count, df_summary, df_reach, top_df, df_nps_over_time, df_sentiment, df_positive_post, df_negative_post,df_most_active_perfile_and_positive_nps, df_most_active_perfile_and_negative_nps, df_entity

	def pushdata(df_post_count, df_summary, df_reach, top_df, df_nps_over_time, df_sentiment, df_positive_post, df_negative_post,df_most_active_perfile_and_positive_nps, df_most_active_perfile_and_negative_nps, df_entity):
		
			engine = create_engine('postgresql://psone_flash_base:psone@kalacitra.in/psone_flash_base')
			df_post_count.to_sql('insta_post_count',engine,if_exists='append',index=False, method="multi")
			df_summary.to_sql('insta_summary',engine,if_exists='append',index=False, method="multi")
			df_reach.to_sql('insta_reach',engine,if_exists='append',index=False, method="multi")
			top_df.to_sql('insta_freq_words',engine,if_exists='append',index=False, method="multi")
			df_nps_over_time.to_sql('insta_nps',engine,if_exists='append',index=False, method="multi")
			df_sentiment.to_sql('insta_sentiment',engine,if_exists='append',index=False, method="multi")
			df_positive_post.to_sql('insta_positive',engine,if_exists='append',index=False, method="multi")
			df_negative_post.to_sql('insta_negative',engine,if_exists='append',index=False, method="multi")
			df_most_active_perfile_and_positive_nps.to_sql('insta_positive_profile',engine,if_exists='append',index=False, method="multi")
			df_most_active_perfile_and_negative_nps.to_sql('insta_negative_profile',engine,if_exists='append',index=False, method="multi")
			df_entity.to_sql('insta_entity',engine,if_exists='append',index=False, method="multi")
			engine.dispose()

	df_insta = fetch(uri,mydb)
	df_post_count, df_summary, df_reach, top_df, df_nps_over_time, df_sentiment, df_positive_post, df_negative_post, df_most_active_perfile_and_positive_nps, df_most_active_perfile_and_negative_nps, df_entity = transform(df_insta)
	pushdata(df_post_count, df_summary, df_reach, top_df, df_nps_over_time, df_sentiment, df_positive_post, df_negative_post,df_most_active_perfile_and_positive_nps, df_most_active_perfile_and_negative_nps, df_entity)

uri = sys.argv[1]
mydb = sys.argv[2]
instagram_analysis(uri,mydb)
