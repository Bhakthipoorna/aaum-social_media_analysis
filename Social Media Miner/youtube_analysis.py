def youtube_analysis(uri,mydb,tag):

	import pandas as pd
	import numpy as np
	import nltk
	import pymongo as pym
	import os
	import tweepy as tw
	import json
	import re
	import pytz
	import datetime
	from configparser import ConfigParser
	from nltk.corpus import stopwords
	from nltk.stem.wordnet import WordNetLemmatizer
	from sklearn.feature_extraction.text import CountVectorizer
	from nltk.sentiment.vader import SentimentIntensityAnalyzer
	sid = SentimentIntensityAnalyzer()
	from dateutil.relativedelta import relativedelta
	import pytz
	IST = pytz.timezone('Asia/Kolkata')
	import psycopg2
	from sqlalchemy import create_engine
	my_date = datetime.date.today()
	year, week, day_of_week = my_date.isocalendar()
	last_date_of_week = pd.to_datetime(datetime.date(year, 1, 1) + relativedelta(weeks=+week))
	last_date_of_week1=str(last_date_of_week)
	last_date_of_week1=last_date_of_week1[0:10]
	last_date_of_week=pd.to_datetime(last_date_of_week1)
	prev_week = pd.to_datetime(datetime.date(year, 1, 1) + relativedelta(weeks=week-1))
	prev_week=str(prev_week)
	prev_week=prev_week[0:10]

	def fetch(uri,mydb,tag):
		
		uri = uri #"mongodb://localhost:27017"
		client= pym.MongoClient(uri)
		mydb = client[mydb] #client['socialmedia']

		if tag=='Teaser':

			youtube_comments= mydb['youtube_video_comments']
			df = pd.DataFrame(list(youtube_comments.find()))

		elif tag=='Ponni':

			youtube_comments= mydb['youtube_video_songs']
			df = pd.DataFrame(list(youtube_comments.find()))

		elif tag=='Chola':

			youtube_comments= mydb['youtube_video_songs']
			df = pd.DataFrame(list(youtube_comments.find()))

		elif tag=='Trailer':

			youtube_comments= mydb['youtube_video_comments_trailer']
			df = pd.DataFrame(list(youtube_comments.find()))

		client.close()

		return df


	def transform(df):

		if tag=='Teaser':

			df=df[~df['comment_id'].isnull()]
			df['language']=df['video_id']
			df['language'].replace({'9PYNkUTMDW0':'Kannada','mQHLkXn_kHU':'Telugu','LYMhbm2ORoc':'Tamil','-xvUWCszQPM':'Hindi','FhD1qCWNp2w':'Malayalam'},inplace=True)
			df=df[df['language'].isin(['Kannada','Tamil','Malayalam','Telugu','Hindi'])]
			df.drop_duplicates('_id',keep='last',inplace=True)

		elif tag=='Ponni':
			df=df[~df['comment_id'].isnull()]
			df=df[df['search_name'].str.contains('Ponni Nadhi - Lyric Video|Kaveri Se Milne - Lyric Video|Ponge Nadhi - Lyric Video')]
			conditionsOnSearchName = [df['search_name'].str.contains('Tamil'),df['search_name'].str.contains('Hindi'),df['search_name'].str.contains('Kannada'),df['search_name'].str.contains('Telugu'),df['search_name'].str.contains('Malayalam')]
			valuesOnLanguage = ['Tamil','Hindi','Kannada','Telugu','Malayalam']
			df['language'] = np.select(conditionsOnSearchName, valuesOnLanguage)
			df.drop_duplicates('_id',keep='last',inplace=True)

		elif tag=='Chola':
			df=df[~df['comment_id'].isnull()]
			df=df[df['search_name'].str.contains('Chola Chola - Lyric Video')]
			conditionsOnSearchName = [df['search_name'].str.contains('Tamil'),df['search_name'].str.contains('Hindi'),df['search_name'].str.contains('Kannada'),df['search_name'].str.contains('Telugu'),df['search_name'].str.contains('Malayalam')]
			valuesOnLanguage = ['Tamil','Hindi','Kannada','Telugu','Malayalam']
			df['language'] = np.select(conditionsOnSearchName, valuesOnLanguage)
			df.drop_duplicates('_id',keep='last',inplace=True)

		elif tag=='Trailer':
			df=df[~df['comment_id'].isnull()]
			conditionsOnSearchName = [df['search_name'].str.contains('Tamil'),df['search_name'].str.contains('Hindi'),df['search_name'].str.contains('Kannada'),df['search_name'].str.contains('Telugu'),df['search_name'].str.contains('Malayalam')]
			valuesOnLanguage = ['Tamil','Hindi','Kannada','Telugu','Malayalam']
			df['language'] = np.select(conditionsOnSearchName, valuesOnLanguage)
			df.drop_duplicates('_id',keep='last',inplace=True)


		df['updated_at']=pd.to_datetime(df['updated_at'])
		df=df[df['updated_at']<'2022-09-22']
		df=df[df['updated_at']>=prev_week]
		df['date']=df['updated_at'].dt.date
		df['date']=pd.to_datetime(df['date'])

		df['comments_text']=df['comments_text'].str.replace('[^A-Za-z]', ' ', flags=re.UNICODE, regex= True)
		df['comments_text']=df['comments_text'].str.lower()

		df['views_count']=pd.to_numeric(df['views_count'], errors='coerce').convert_dtypes()
		df['like_count']=pd.to_numeric(df['like_count'], errors='coerce').convert_dtypes()
		df['comment_count']=pd.to_numeric(df['comment_count'], errors='coerce').convert_dtypes()

		df['views_count'].fillna(0,inplace=True)
		df['like_count'].fillna(0,inplace=True)
		df['comment_count'].fillna(0,inplace=True)

		df_likes_views_comments=pd.DataFrame(df.groupby(['language']).agg({"like_count":np.max,"views_count":np.max,"comment_count":np.max})).reset_index()
		df_likes_views_comments['date']=last_date_of_week
		df_likes_views_comments['date']=pd.to_datetime(df_likes_views_comments['date'])
		df_likes_views_comments['tag']=tag
		df_likes=df_likes_views_comments[['date','tag','language','like_count']]
		df_views=df_likes_views_comments[['date','tag','language','views_count']]
		df_comments=df_likes_views_comments[['date','tag','language','comment_count']]


		df_likes_views_comments=pd.DataFrame(df.groupby(['language','date']).agg({"like_count":np.max,"views_count":np.max,"comment_count":np.max})).reset_index()
		df_likes_views_comments['tag']=tag
		df_likes_overtime=df_likes_views_comments[['date','tag','language','like_count']]
		df_views_overtime=df_likes_views_comments[['date','tag','language','views_count']]
		df_comments_overtime=df_likes_views_comments[['date','tag','language','comment_count']]
		df_likes_overtime=df_likes_overtime.sort_values('date')
		df_views_overtime=df_views_overtime.sort_values('date')
		df_comments_overtime=df_comments_overtime.sort_values('date')
        
		df_reach=(pd.merge(df_views_overtime,df_likes_overtime,on=['date','tag','language'])).merge(df_comments_overtime,on=['date','tag','language'])
		df_reach['reach']=df_reach['views_count']+df_reach['like_count']+df_reach['comment_count']
		df_reach=df_reach[['date','tag','language','reach']]

		stop_words = set(stopwords.words("english"))
		new_words = ['amp','co', 'http','youtube','http','www','com','href','result ','search','query','result', 'br', 'channel', 'audience','hai', 'sir', 'youtuber','video','film','please','language','ki','ka','p','se','movie','mera','help']
		stop_words = stop_words.union(new_words)

		corp = []
		for i in range(0, df.shape[0]):
			#Remove punctuations
			text = re.sub('[^a-zA-Z]', ' ', str(df['comments_text'].iloc[i])) 
			text=text.lower()
			##Convert to list from string
			text = text.split()
			##Lemmatizing
			lm = WordNetLemmatizer() 
			text = [lm.lemmatize(word) for word in text if not word in stop_words] 
			text = " ".join(text)
			corp.append(text)

		#Removing stopwords from the comments_text column
		df['comments_text']=df['comments_text'].astype('str')
		df["comments_text"]=df["comments_text"].apply(lambda words: ' '.join(word.lower() for word in words.split() if word not in stop_words))

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
		top_df['tag']=tag
		drop_words=['please','others','ponniyin','selvan','ponniyin selvan']
		top_df=top_df[~top_df['word'].isin(drop_words)]
		top_df=top_df.head(5)
		top_df=top_df[['date','tag','word','freq']]
        

		df['comments_text']=df['comments_text'].astype('str')
		df['senti']=df['comments_text'].apply(sid.polarity_scores)

		df['compsenti'] = df['senti'].apply(lambda x: x['compound'])
		df['possenti'] = df['senti'].apply(lambda x: x['pos'])
		df['negsenti'] = df['senti'].apply(lambda x: x['neg'])

		df=df.set_index('updated_at')
		df['nps']= df['possenti']-df['negsenti']
		df_nps = pd.DataFrame(df.groupby(['language','date'])['nps'].mean()).reset_index()
		df_nps['tag']=tag
		df_nps=df_nps[['date','tag','language','nps']]

		#most positive comment
		df_positive=pd.DataFrame(df.groupby(by= ['language','comments_text'])[ 'possenti'].mean().sort_values(ascending = False)).reset_index()

		df_positive=df_positive.sort_values(['language','possenti'],ascending=False)
		df_positive=df_positive.groupby('language').head(5)
		df_positive_telugu=df_positive[df_positive['language']=='Telugu']
		df_positive_tamil=df_positive[df_positive['language']=='Tamil']
		df_positive_hindi=df_positive[df_positive['language']=='Hindi']
		df_positive_kannada=df_positive[df_positive['language']=='Kannada']
		df_positive_malayalam=df_positive[df_positive['language']=='Malayalam']

		df_telugu_positive=list(df_positive_telugu['comments_text'])
		df_tamil_positive=list(df_positive_tamil['comments_text'])
		df_hindi_positive=list(df_positive_hindi['comments_text'])
		df_kannada_positive=list(df_positive_kannada['comments_text'])
		df_malayalam_positive=list(df_positive_malayalam['comments_text'])

		telugu_positive=pd.DataFrame([['Telugu',df_telugu_positive]])
		tamil_positive=pd.DataFrame([['Tamil',df_tamil_positive]])
		hindi_positive=pd.DataFrame([['Hindi',df_hindi_positive]])
		kannada_positive=pd.DataFrame([['Kannada',df_kannada_positive]])
		malayalam_positive=pd.DataFrame([['Malayalam',df_malayalam_positive]])

		df_positive=pd.concat([telugu_positive,tamil_positive,hindi_positive,kannada_positive,malayalam_positive])
		df_positive.columns=['language','positive_words']
		df_positive['date']=last_date_of_week
		#df_positive['date']='2022-08-27'
		df_positive['date']=pd.to_datetime(df_positive['date'])
		df_positive['tag']=tag
		df_positive=df_positive[['date','tag','language','positive_words']] # Language

		df_negative=pd.DataFrame(df.groupby(by= ['language','comments_text'])[ 'negsenti'].mean().sort_values(ascending = False)).reset_index()

		df_negative=df_negative.sort_values(['language','negsenti'],ascending=False)
		df_negative=df_negative.groupby('language').head(5)
		df_negative_telugu=df_negative[df_negative['language']=='Telugu']
		df_negative_tamil=df_negative[df_negative['language']=='Tamil']
		df_negative_hindi=df_negative[df_negative['language']=='Hindi']
		df_negative_kannada=df_negative[df_negative['language']=='Kannada']
		df_negative_malayalam=df_negative[df_negative['language']=='Malayalam']

		df_negative_telugu=list(df_negative_telugu['comments_text'])
		df_negative_tamil=list(df_negative_tamil['comments_text'])
		df_negative_hindi=list(df_negative_hindi['comments_text'])
		df_negative_kannada=list(df_negative_kannada['comments_text'])
		df_negative_malayalam=list(df_negative_malayalam['comments_text'])

		telugu_negative=pd.DataFrame([['Telugu',df_negative_telugu]])
		tamil_negative=pd.DataFrame([['Tamil',df_negative_tamil]])
		hindi_negative=pd.DataFrame([['Hindi',df_negative_hindi]])
		kannada_negative=pd.DataFrame([['Kannada',df_negative_kannada]])
		malayalam_negative=pd.DataFrame([['Malayalam',df_negative_malayalam]])

		df_negative=pd.concat([telugu_negative,tamil_negative,hindi_negative,kannada_negative,malayalam_negative])
		df_negative.columns=['language','negative_words']
		df_negative['date']=last_date_of_week
		#df_negative['date']='2022-08-27'
		df_negative['date']=pd.to_datetime(df_negative['date'])
		df_negative['tag']=tag
		df_negative=df_negative[['date','tag','language','negative_words']] # Language
		df_positive_negative=pd.merge(df_positive,df_negative,on='language')
		df_positive_negative=pd.merge(df_positive,df_negative,on=['date','tag','language'])


		df_personwise_count=pd.DataFrame(df['author_display_name'].value_counts()).reset_index()
		df_personwise_count.columns=['person_name','comments_count']
		df_nps_per_person=pd.DataFrame(df.groupby('author_display_name')['nps'].mean()).reset_index()
		df_nps_per_person.columns=['person_name','nps']
		df_nps_per_person=df_nps_per_person.sort_values('nps')
		df_nps_per_person=pd.merge(df_personwise_count,df_nps_per_person,on='person_name')
		df_positive_profile=df_nps_per_person.sort_values(['comments_count','nps'],ascending=False)
		df_positive_profile=df_positive_profile.head(10)
		df_positive_profile['date']=last_date_of_week
		df_positive_profile['date']=pd.to_datetime(df_positive_profile['date'])
		df_positive_profile['tag']=tag
		df_positive_profile=df_positive_profile[['date','tag','person_name','comments_count','nps']]

		df_negative_profile=df_nps_per_person.sort_values(['comments_count'],ascending=False)
		df_negative_profile=df_negative_profile[df_negative_profile['nps']<0]
		df_negative_profile=df_negative_profile.head(10)
		df_negative_profile['date']=last_date_of_week
		df_negative_profile['date']=pd.to_datetime(df_negative_profile['date'])
		df_negative_profile['tag']=tag
		df_negative_profile=df_negative_profile[['date','tag','person_name','comments_count','nps']]


		df['vfx'] = df['comments_text'].str.contains('vfx',case = False)
		vfx=pd.DataFrame(df['vfx'].resample('1d').sum().reset_index())
		vfx.rename(columns={'updated_at':'date'},inplace=True)
		vfx['date']=vfx['date'].dt.date
		vfx['date']=pd.to_datetime(vfx['date'])
		vfx_daywise=pd.DataFrame(df.groupby(by='date').vfx.count()).reset_index()
		vfx_daywise['date']=pd.to_datetime(vfx_daywise['date'])
		merged_vfx= pd.merge(vfx, vfx_daywise, on='date', how= 'outer' )
		merged_vfx['att_sc']= merged_vfx['vfx_x']/merged_vfx['vfx_y']
		merged_vfx.fillna(0,inplace=True)
		merged_vfx=merged_vfx[['date','att_sc']]
		merged_vfx['entity']='vfx'

		df['grand'] = df['comments_text'].str.contains('grand',case = False)
		grand=pd.DataFrame(df['grand'].resample('1d').sum().reset_index())
		grand.rename(columns={'updated_at':'date'},inplace=True)
		grand['date']=grand['date'].dt.date
		grand['date']=pd.to_datetime(grand['date'])
		grand_daywise=pd.DataFrame(df.groupby(by='date').grand.count()).reset_index()
		grand_daywise['date']=pd.to_datetime(grand_daywise['date'])
		merged_grand= pd.merge(grand, grand_daywise, on='date', how= 'outer' )
		merged_grand['att_sc']= merged_grand['grand_x']/merged_grand['grand_y']
		merged_grand.fillna(0,inplace=True)
		merged_grand=merged_grand[['date','att_sc']]
		merged_grand['entity']='grand'

		df['director'] = df['comments_text'].str.contains('director',case = False)
		director=pd.DataFrame(df['director'].resample('1d').sum().reset_index())
		director.rename(columns={'updated_at':'date'},inplace=True)
		director['date']=director['date'].dt.date
		director['date']=pd.to_datetime(director['date'])
		director_daywise=pd.DataFrame(df.groupby(by='date').director.count()).reset_index()
		director_daywise['date']=pd.to_datetime(director_daywise['date'])
		merged_director= pd.merge(director, director_daywise, on='date', how= 'outer' )
		merged_director['att_sc']= merged_director['director_x']/merged_director['director_y']
		merged_director.fillna(0,inplace=True)
		merged_director=merged_director[['date','att_sc']]
		merged_director['entity']='director'


		df['music'] = df['comments_text'].str.contains('music',case = False)
		music=pd.DataFrame(df['music'].resample('1d').sum().reset_index())
		music.rename(columns={'updated_at':'date'},inplace=True)
		music['date']=music['date'].dt.date
		music['date']=pd.to_datetime(music['date'])
		music_daywise=pd.DataFrame(df.groupby(by='date').music.count()).reset_index()
		music_daywise['date']=pd.to_datetime(music_daywise['date'])
		merged_music= pd.merge(music, music_daywise, on='date', how= 'outer' )
		merged_music['att_sc']= merged_music['music_x']/merged_music['music_y']
		merged_music.fillna(0,inplace=True)
		merged_music=merged_music[['date','att_sc']]
		merged_music['entity']='music'


		df['camera'] = df['comments_text'].str.contains('camera',case = False)
		camera=pd.DataFrame(df['camera'].resample('1d').sum().reset_index())
		camera.rename(columns={'updated_at':'date'},inplace=True)
		camera['date']=camera['date'].dt.date
		camera['date']=pd.to_datetime(camera['date'])
		camera_daywise=pd.DataFrame(df.groupby(by='date').camera.count()).reset_index()
		camera_daywise['date']=pd.to_datetime(camera_daywise['date'])
		merged_camera= pd.merge(camera, camera_daywise, on='date', how= 'outer' )
		merged_camera['att_sc']= merged_camera['camera_x']/merged_camera['camera_y']
		merged_camera.fillna(0,inplace=True)
		merged_camera=merged_camera[['date','att_sc']]
		merged_camera['entity']='camera'


		df['sets'] = df['comments_text'].str.contains('sets',case = False)
		sets=pd.DataFrame(df['sets'].resample('1d').sum().reset_index())
		sets.rename(columns={'updated_at':'date'},inplace=True)
		sets['date']=sets['date'].dt.date
		sets['date']=pd.to_datetime(sets['date'])
		sets_daywise=pd.DataFrame(df.groupby(by='date').sets.count()).reset_index()
		sets_daywise['date']=pd.to_datetime(sets_daywise['date'])
		merged_sets= pd.merge(sets, sets_daywise, on='date', how= 'outer' )
		merged_sets['att_sc']= merged_sets['sets_x']/merged_sets['sets_y']
		merged_sets.fillna(0,inplace=True)
		merged_sets=merged_sets[['date','att_sc']]
		merged_sets['entity']='sets'


		df['history'] = df['comments_text'].str.contains('history',case = False)
		history=pd.DataFrame(df['history'].resample('1d').sum().reset_index())
		history.rename(columns={'updated_at':'date'},inplace=True)
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
		df_entity['tag']=tag
		df_entity=df_entity.sort_values('date')

		return df_views, df_likes, df_comments, df_reach, df_views_overtime, df_likes_overtime, df_comments_overtime, top_df, df_nps, df_positive_negative, df_positive_profile, df_negative_profile, df_entity


	def pushdata(df_views, df_likes, df_comments, df_reach, df_views_overtime, df_likes_overtime, df_comments_overtime, top_df, df_nps, df_positive_negative, df_positive_profile, df_negative_profile, df_entity):
		engine = create_engine('postgresql://psone_flash_base:psone@kalacitra.in/psone_flash_base')

		#df_views.to_sql('yt_views',engine,if_exists='append',index=False, method="multi")
		#df_likes.to_sql('yt_likes',engine,if_exists='append',index=False, method="multi")
		#df_comments.to_sql('yt_comments',engine,if_exists='append',index=False, method="multi")
		df_reach.to_sql('yt_reach',engine,if_exists='append',index=False, method="multi")
		df_views_overtime.to_sql('yt_views_overtime',engine,if_exists='append',index=False, method="multi")
		df_likes_overtime.to_sql('yt_likes_overtime',engine,if_exists='append',index=False, method="multi")
		df_comments_overtime.to_sql('yt_comments_overtime',engine,if_exists='append',index=False, method="multi")
		#top_df.to_sql('yt_freq_words',engine,if_exists='append',index=False, method="multi")
		df_nps.to_sql('yt_nps',engine,if_exists='append',index=False, method="multi")
		#df_positive_negative.to_sql('yt_positive_negative',engine,if_exists='append',index=False, method="multi")
		#df_positive_profile.to_sql('yt_nps_per_person',engine,if_exists='append',index=False, method="multi")
		#df_negative_profile.to_sql('yt_nps_per_person',engine,if_exists='append',index=False, method="multi")
		df_entity.to_sql('yt_entity',engine,if_exists='append',index=False, method="multi")
		engine.dispose()

	df = fetch(uri,mydb,tag)
	df_views, df_likes, df_comments, df_reach, df_views_overtime, df_likes_overtime, df_comments_overtime, top_df, df_nps, df_positive_negative, df_positive_profile, df_negative_profile, df_entity = transform(df)
	#pushdata(df_views, df_likes, df_comments, df_reach, df_views_overtime, df_likes_overtime, df_comments_overtime, top_df, df_nps, df_positive_negative, df_positive_profile, df_negative_profile, df_entity)

	return df_views,df_likes,df_comments,df_reach, df_views_overtime,df_likes_overtime,df_comments_overtime,top_df,df_nps, df_positive_negative, df_positive_profile, df_negative_profile, df_entity
df_views, df_likes, df_comments, df_reach, df_views_overtime, df_likes_overtime, df_comments_overtime, top_df, df_nps, df_positive_negative, df_positive_profile, df_negative_profile, df_entity = youtube_analysis(uri,mydb,tag)