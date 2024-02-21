
def youtube_analysis_daily(uri,mydb,tag):

	import pandas as pd
	import numpy as np
	import nltk
	import pymongo as pym
	import os
	import tweepy as tw
	import json
	import re
	import pytz
	import psycopg2
	import datetime
	from sqlalchemy import create_engine
	from configparser import ConfigParser
	from nltk.corpus import stopwords
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
	prior_2days=datetime.strftime(datetime.now(IST) - timedelta(2), '%Y-%m-%d')

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

		elif tag=='Ratchasa Maamaney':
			youtube_comments= mydb['youtube_video_songs']
			df = pd.DataFrame(list(youtube_comments.find()))

		elif tag=='Alaikadal':
			youtube_comments= mydb['youtube_video_songs']
			df = pd.DataFrame(list(youtube_comments.find()))

		elif tag=='Sol':
			youtube_comments= mydb['youtube_video_songs']
			df = pd.DataFrame(list(youtube_comments.find()))

		elif tag=='Alaikadal-Glimpse':
			youtube_comments= mydb['youtube_video_comments_glimpse']
			df = pd.DataFrame(list(youtube_comments.find()))

		elif tag=='Sol-Glimpse':
			youtube_comments= mydb['youtube_video_comments_glimpse']
			df = pd.DataFrame(list(youtube_comments.find()))

		elif tag=='Chola-BTS':
			youtube_comments= mydb['youtube_video_comments_bts']
			df = pd.DataFrame(list(youtube_comments.find()))

		elif tag=='Promo':
			youtube_comments= mydb['youtube_video_comments_promo']
			df = pd.DataFrame(list(youtube_comments.find()))

		elif tag=='Review':
			youtube_comments= mydb['youtube_video_comments_reviews']
			youtube_comments_kannada= mydb['youtube_video_comments_reviews_kannada']
			youtube_comments_telugu= mydb['youtube_video_comments_reviews_telugu']
			df1 = pd.DataFrame(list(youtube_comments.find()))
			df2 = pd.DataFrame(list(youtube_comments_kannada.find()))
			df3 = pd.DataFrame(list(youtube_comments_telugu.find()))
			df=pd.concat([df1,df2,df3],ignore_index=False)
		elif tag=='Competitor-NaaneVaruvean':
			youtube_comments= mydb['youtube_video_comments_reviews_competitor']
			df = pd.DataFrame(list(youtube_comments.find()))
		elif tag=='Celebrity-Talks':
			youtube_comments= mydb['youtube_video_comments_celebrity_talk']
			df = pd.DataFrame(list(youtube_comments.find()))


		client.close()

		conn = psycopg2.connect(database="psone_flash_base", user='psone_flash_base', password='psone', host='kalacitra.in')  
		cursor = conn.cursor()

		tb_views=pd.read_sql("select * from yt_views_overtime where date ='{}' ".format(prior_2days),conn)
		tb_likes=pd.read_sql("select * from yt_likes_overtime where date ='{}'".format(prior_2days),conn)
		tb_comments=pd.read_sql("select * from yt_comments_overtime where date ='{}'".format(prior_2days),conn)

		cursor.close()
		conn.close()

		return df, tb_views, tb_likes, tb_comments


	def transform(df, tb_views, tb_likes, tb_comments):

		tb_views=tb_views[tb_views['tag']==tag]
		tb_likes=tb_likes[tb_likes['tag']==tag]
		tb_comments=tb_comments[tb_comments['tag']==tag]

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

		elif tag=='Ratchasa Maamaney':
			df=df[~df['comment_id'].isnull()]
			df['language']=df['video_id']
			df['language'].replace({'o2CIBEkVgR0':'Tamil','Esu2_-3Tx9I':'Hindi','p9pEOzhPIFo':'Telugu','Uy-jIbppVOA':'Kannada','fh4kZ8AZ2Qo':'Malayalam'},inplace=True)
			df=df[df['language'].isin(['Kannada','Tamil','Malayalam','Telugu','Hindi'])]
			df.drop_duplicates('_id',keep='last',inplace=True)

		elif tag=='Alaikadal':
			df=df[~df['comment_id'].isnull()]
			df['language']=df['video_id']
			df['language'].replace({'bh6et8Ko200':'Tamil','N0EYZhWB-eg':'Telugu','LvtueH0pVO8':'Malayalam','yWoP4G0-N3E':'Hindi','-EbGj_5k654':'Kannada'},inplace=True)
			df=df[df['language'].isin(['Kannada','Tamil','Malayalam','Telugu','Hindi'])]
			df.drop_duplicates('_id',keep='last',inplace=True)

		elif tag=='Sol':
			df=df[~df['comment_id'].isnull()]
			df['language']=df['video_id']
			df['language'].replace({'S2PPXrFrLI8':'Tamil','LLidly-le-k':'Hindi','h_R9EwMqvV4':'Kannada','KaMEoz64RWk':'Telugu','AkASC0Zyh8Y':'Malayalam'},inplace=True)
			df=df[df['language'].isin(['Kannada','Tamil','Malayalam','Telugu','Hindi'])]

		elif tag=='Alaikadal-Glimpse':
			df=df[~df['comment_id'].isnull()]
			df['language']=df['video_id']
			df['language'].replace({'UeanT-X3DeU':'Tamil','RIeYol-72qo':'Hindi','WLmdmo44Dl0':'Telugu','gFuiVazURTY':'Kannada','TybidKkWPEc':'Malayalam'},inplace=True)
			df=df[df['language'].isin(['Kannada','Tamil','Malayalam','Telugu','Hindi'])]

		elif tag=='Sol-Glimpse':
			df=df[~df['comment_id'].isnull()]
			df['language']=df['video_id']
			df['language'].replace({'xyKRxp1tmLs':'Tamil','se1eLKr-9hg':'Hindi','fFs4McnDlSc':'Telugu','BnIKEqUgHGI':'Kannada','BWWXLpdJvAE':'Malayalam'},inplace=True)
			df=df[df['language'].isin(['Kannada','Tamil','Malayalam','Telugu','Hindi'])]

		elif tag=='Chola-BTS':
			df=df[~df['comment_id'].isnull()]
			df['language']=df['video_id']
			df['language'].replace({'a39JJJj70gs':'Tamil','3OFygppvj1w':'Hindi','kHU6wL9LIJo':'Telugu','XL7cTuXA6GE':'Kannada','JwHaZO0TXa0':'Malayalam'},inplace=True)
			df=df[df['language'].isin(['Kannada','Tamil','Malayalam','Telugu','Hindi'])]

		elif tag=='Promo':
			df=df[~df['comment_id'].isnull()]		
			df=df[~df['comment_id'].isnull()]
			df['language']=df['video_id']
			df['language'].replace({'4t88s6UGCBM':'Pandys-Intro','suFLcx0nhMQ':'Promo1','FkefqN-tEjA':'Promo2','Ojfj7fupt8M':'Promo3'},inplace=True)
			df=df[df['language'].isin(['Pandys-Intro','Promo1','Promo2','Promo3'])]

		elif tag=='Review':
			df=df[~df['comment_id'].isnull()]
			df['language']=df['video_id']
			df['language'].replace({'oBpXLhphoV0':'ReviewPunjabiThikana','kIT70IT6aEY':'ReviewMaahiTalkies', '9u3RGRDTRIk':'TrendingToday', 'RPt7G-f4mAU':'BenchMarkKannada','xrnONtewW8g':'ReviewCornerKannada','5kiTnaFhgAQ':'CinibuzzKannada', 'KY55xWZbQ84':'MahidarVibesTelugu','8qOzL4iuc_I':'MovieMattersTelugu','3M9R5duRazM':'TeluguOne','xaqrPCg3ooc':'ATVTelugu'},inplace=True)
			df=df[df['language'].isin(['ReviewPunjabiThikana','ReviewMaahiTalkies','TrendingToday','BenchMarkKannada','ReviewCornerKannada','CinibuzzKannada','MahidarVibesTelugu','MovieMattersTelugu','TeluguOne','ATVTelugu'])]

		elif tag=='Competitor-NaaneVaruvean':
			df=df[~df['comment_id'].isnull()]
			df['language']=df['video_id']
			df['language'].replace({'Asvf5tiKHAk':'TamilTalkies','GZJqcrKAM4k':'TamilCinemaReview','m0J3gkQ--gs':'TodayTrending','rqQEuYtY_rc':'TamilGlitz1','DKnvJHQDelI':'TamilGlitz2','eAWzQXskxbw':'TamilGlitz3','7iTRB6zY5Rk':'BehindwoodTV'},inplace=True)
			df=df[df['language'].isin(['TamilTalkies','TamilCinemaReview','TodayTrending','TamilGlitz1','TamilGlitz2','TamilGlitz3','BehindwoodTV'])]

		elif tag=='Celebrity-Talks':
			df=df[~df['comment_id'].isnull()]
			df['language']=df['video_id']
			df['language'].replace({'RJH4kHDKQR0':'AudioLaunch','QfBXwoj9Dfs':'FunChat-Tamil','1tubJwKhqek':'FunChat-Telugu','eZgosGRx1sM':'FunChat-Kannada','j-lqRbxOwQw':'FunChat-promo-Kannada','Gy3QeEV_J3o':'FunChat-promo-Telugu','BHYHv7_jaW8':'Parthiban-Sarathkumar','j-FwODZMCog':'Shobita','wwmzdpkJqyk':'Trisha','JIZ2zhlbO_Y':'Jayam Ravi','Ucf7dFeroQQ':'subakaran','56BYWeuGBPE':'Rajinikanth','oUjCpuSYlRE':'Jayaram','qzjjgSo6kTE':'Karthi','8S09-mhiVLI':'Shankar','HAUl4ZT2ttQ':'Aishwarya Lekshmi'},inplace=True)
			df=df[df['language'].isin(['AudioLaunch','FunChat-Tamil','FunChat-Telugu','FunChat-Kannada','FunChat-promo-Kannada','FunChat-promo-Telugu','Parthiban-Sarathkumar','Shobita','Trisha','Jayam Ravi','Rajinikanth','Jayaram','Karthi','Shankar','Aishwarya Lekshmi'])]

		df['updated_at']=pd.to_datetime(df['updated_at'])
		df=df[df['updated_at']<Today]
		df=df[df['updated_at']>=yesterday]

		if len(df)<1:

			df_nps=pd.DataFrame(columns=['date','tag','language','nps','possenti','negsenti'])
			df_entity=pd.DataFrame(columns=['date','tag','percentage'])
			df_reach=pd.DataFrame(columns=['date','tag','language','reach'])
			df_sentiment=pd.DataFrame(columns=['date','tag','language','mertic','score'])
			df_views_overtime=tb_views
			df_views_overtime['date']=yesterday
			df_likes_overtime=tb_likes
			df_likes_overtime['date']=yesterday
			df_comments_overtime=tb_comments
			df_comments_overtime['date']=yesterday


		else :

			df[["updated_at"]]=df[["updated_at"]].apply(pd.to_datetime, errors='coerce') 
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

			df_likes_views_comments=pd.DataFrame(df.groupby(['language','date']).agg({"like_count":np.max,"views_count":np.max,"comment_count":np.max})).reset_index()
			df_likes_views_comments['tag']=tag
			df_likes_overtime=df_likes_views_comments[['date','tag','language','like_count']]
			df_views_overtime=df_likes_views_comments[['date','tag','language','views_count']]
			df_views_overtime=pd.merge(df_views_overtime, tb_views, on=['tag','language'],how='outer')
			df_views_overtime=df_views_overtime[['date_x','date_y','tag','language','views_count_x','views_count_y']]
			df_views_overtime['views_count_x']=df_views_overtime['views_count_x'].fillna(df_views_overtime['views_count_y'])
			df_views_overtime=df_views_overtime[['date_x','tag','language','views_count_x']]
			df_views_overtime=df_views_overtime.ffill()
			df_views_overtime.columns=['date','tag','language','views_count']

			df_comments_overtime=df_likes_views_comments[['date','tag','language','comment_count']]
			df_likes_overtime=df_likes_overtime.sort_values('date')
			df_views_overtime=df_views_overtime.sort_values('date')
			df_comments_overtime=df_comments_overtime.sort_values('date')

			df_likes_overtime=pd.merge(df_likes_overtime, tb_likes, on=['tag','language'],how='outer')
			df_likes_overtime=df_likes_overtime[['date_x','date_y','tag','language','like_count_x','like_count_y']]
			df_likes_overtime['like_count_x']=df_likes_overtime['like_count_x'].fillna(df_likes_overtime['like_count_y'])
			df_likes_overtime=df_likes_overtime[['date_x','tag','language','like_count_x']]
			df_likes_overtime=df_likes_overtime.ffill()
			df_likes_overtime.columns=['date','tag','language','like_count']

			df_comments_overtime=pd.merge(df_comments_overtime, tb_comments, on=['tag','language'],how='outer')
			df_comments_overtime=df_comments_overtime[['date_x','date_y','tag','language','comment_count_x','comment_count_y']]
			df_comments_overtime['comment_count_x']=df_comments_overtime['comment_count_x'].fillna(df_comments_overtime['comment_count_y'])
			df_comments_overtime=df_comments_overtime[['date_x','tag','language','comment_count_x']]
			df_comments_overtime=df_comments_overtime.ffill()
			df_comments_overtime.columns=['date','tag','language','comment_count']

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


			if len(df['comments_text'])>10:
				cv=CountVectorizer(max_df=1, stop_words=stop_words, ngram_range=(1,2), min_df=0)
				X=cv.fit_transform(corp)
				vector = cv.transform(corp)
			else:
				pass
			#Most frequently occuring words
			def get_top_n_words(corpus, n=None):
				vec = cv.fit(corp)
				bag_of_words = vec.transform(corp)
				sum_words = bag_of_words.sum(axis=0) 
				words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
				words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
				return words_freq[:n]

			df['comments_text']=df['comments_text'].astype('str')
			df['senti']=df['comments_text'].apply(sid.polarity_scores)

			df['compsenti'] = df['senti'].apply(lambda x: x['compound'])
			df['possenti'] = df['senti'].apply(lambda x: x['pos'])
			df['negsenti'] = df['senti'].apply(lambda x: x['neg'])
			df[['negsenti','possenti']]=df[['negsenti','possenti']]*100

			df=df.set_index('updated_at')
			df['nps']= df['possenti']-df['negsenti']
			df_nps = pd.DataFrame(df.groupby(['language','date'])['nps','possenti','negsenti'].mean()).reset_index()
			df_nps['tag']=tag
			df_nps=df_nps[['date','tag','language','nps','possenti','negsenti']]

			nps=df_nps[['date','tag','language','nps']]
			nps['metric']='nps'
			nps.rename(columns={'nps':'score'},inplace=True)
			possenti=df_nps[['date','tag','language','possenti']]
			possenti['metric']='possenti'
			possenti.rename(columns={'possenti':'score'},inplace=True)
			negsenti=df_nps[['date','tag','language','negsenti']]
			negsenti['metric']='negsenti'
			negsenti.rename(columns={'negsenti':'score'},inplace=True)
			df_sentiment=pd.concat([nps,possenti,negsenti],ignore_index=False)
			df_sentiment=df_sentiment.fillna(0)
			df_sentiment=df_sentiment[['date','tag','language','metric','score']]
			df_sentiment['metric'].replace({'possenti':'positive','negsenti':'negative'},inplace=True)

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
			merged_vfx=merged_vfx[['date','att_sc','vfx_x','vfx_y']]
			merged_vfx.rename(columns={'vfx_x':'entity_count','vfx_y':'total_comments'},inplace=True)
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
			merged_grand=merged_grand[['date','att_sc','grand_x','grand_y']]
			merged_grand.rename(columns={'grand_x':'entity_count','grand_y':'total_comments'},inplace=True)
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
			merged_director=merged_director[['date','att_sc','director_x','director_y']]
			merged_director.rename(columns={'director_x':'entity_count','director_y':'total_comments'},inplace=True)
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
			merged_music=merged_music[['date','att_sc','music_x','music_y']]
			merged_music.rename(columns={'music_x':'entity_count','music_y':'total_comments'},inplace=True)
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
			merged_camera=merged_camera[['date','att_sc','camera_x','camera_y']]
			merged_camera.rename(columns={'camera_x':'entity_count','camera_y':'total_comments'},inplace=True)
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
			merged_sets=merged_sets[['date','att_sc','sets_x','sets_y']]
			merged_sets.rename(columns={'sets_x':'entity_count','sets_y':'total_comments'},inplace=True)
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
			merged_history=merged_history[['date','att_sc','history_x','history_y']]
			merged_history.rename(columns={'history_x':'entity_count','history_y':'total_comments'},inplace=True)
			merged_history['entity']='history'

			df_entity=pd.concat([merged_vfx,merged_grand,merged_director,merged_music,merged_camera,merged_sets,merged_history],ignore_index=False)
			df_entity.rename(columns={'att_sc':'percentage'},inplace=True)
			df_entity['percentage'] = df_entity['percentage']*100
			df_entity['tag']=tag
			df_entity=df_entity.sort_values('date')

		return df_reach, df_views_overtime, df_likes_overtime, df_comments_overtime, df_nps, df_sentiment, df_entity

	def pushdata(df_reach, df_views_overtime, df_likes_overtime, df_comments_overtime, df_nps, df_sentiment, df_entity):
		engine = create_engine('postgresql://psone_flash_base:psone@kalacitra.in/psone_flash_base')
		df_reach.to_sql('yt_reach',engine,if_exists='append',index=False, method="multi")
		df_views_overtime.to_sql('yt_views_overtime',engine,if_exists='append',index=False, method="multi")
		df_likes_overtime.to_sql('yt_likes_overtime',engine,if_exists='append',index=False, method="multi")
		df_comments_overtime.to_sql('yt_comments_overtime',engine,if_exists='append',index=False, method="multi")
		df_nps.to_sql('yt_nps',engine,if_exists='append',index=False, method="multi")
		df_sentiment.to_sql('yt_sentiment',engine,if_exists='append',index=False, method="multi")
		df_entity.to_sql('yt_entity',engine,if_exists='append',index=False, method="multi")
		engine.dispose()

	df, tb_views, tb_likes, tb_comments = fetch(uri,mydb,tag)
	df_reach, df_views_overtime, df_likes_overtime, df_comments_overtime, df_nps, df_sentiment, df_entity = transform(df, tb_views, tb_likes, tb_comments)
	pushdata(df_reach, df_views_overtime, df_likes_overtime, df_comments_overtime, df_nps, df_sentiment, df_entity)

