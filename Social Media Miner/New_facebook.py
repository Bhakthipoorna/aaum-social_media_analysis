def FacebookAnalysis(uri,mydb):
	
	import pandas as pd
	import numpy as np
	import nltk
	import os
	import json
	import re
	import pymongo as pym
	from nltk.corpus import stopwords
	from nltk.stem.wordnet import WordNetLemmatizer 
	from sklearn.feature_extraction.text import TfidfVectorizer
	from nltk.sentiment.vader import SentimentIntensityAnalyzer
	sid = SentimentIntensityAnalyzer()
	import datetime
	import pytz
	IST = pytz.timezone('Asia/Kolkata')
	from dateutil.relativedelta import relativedelta
	from nltk.sentiment.vader import SentimentIntensityAnalyzer
	import psycopg2
	from sqlalchemy import create_engine
	my_date = datetime.date.today() # if date is 01/01/2018
	year, week, day_of_week = my_date.isocalendar()
	last_date_of_week = pd.to_datetime(datetime.date(year, 1, 1) + relativedelta(weeks=+week))
	last_date_of_week=str(last_date_of_week)
	last_date_of_week=last_date_of_week[0:10]

	def fetch(uri,mydb):

		uri = uri #"mongodb://localhost:27017"
		client= pym.MongoClient(uri)
		mydb = client[mydb] #client['Manual_Data_Push_For_FB']
		fb_comments= mydb['Manual_Data_Push_For_FB_Comment_Scapper']
		df1= pd.json_normalize(fb_comments.find())
		client.close()

		df2_excel=pd.ExcelFile('FB_lookup_24Aug.xlsx',engine='openpyxl')
		df2 = pd.read_excel(df2_excel, 'Relavant')

		return df1, df2

	def transform(df1,df2):
		
		df2=df2[['Url']]
		df1['Url']=df1['Url'].str.strip()
		df2['Url']=df2['Url'].str.strip()

		df=df1[df1['Url'].isin(df2['Url'])]
		df.drop_duplicates(keep='last',inplace=True)

		df_post_count=pd.DataFrame([[last_date_of_week,df['Url'].nunique()]],columns=["date","posts_count"])
		df_comment_count=pd.DataFrame([[last_date_of_week,df.shape[0]]],columns=["date","comments_count"])
		df_user_count=pd.DataFrame([[last_date_of_week,df['Person Name'].nunique()]],columns=["date","user_count"])

		df['Likes Counts']=df['Likes Counts'].str.replace("[","")
		df['Likes Counts']=df['Likes Counts'].str.replace("]","")
		df['Likes Counts']=df['Likes Counts'].str.replace("'","")
		df['Likes Counts']=df['Likes Counts'].apply(lambda x: x.split('\n')[0])
		df['Likes Counts']=df['Likes Counts'].apply(lambda x: x.split('\\n')[0]) # to remove duplicated like counts (ex: 1.4K\n1.4K)

		# converting short form like to numeric (ex: 1.2K to 1200)
		df['Likes Counts']=df['Likes Counts'].str.strip()
		df['Likes Counts']=df['Likes Counts'].str[-15:]
		df['Likes']=df['Likes Counts'].str.replace('[^0-9KM.]',' ', flags=re.UNICODE) #(K for thousand and M for million)
		df_likes_num=df[~df['Likes'].str.contains('K|M')] # video likes in numbers
		df_likes_short_num=df[df['Likes'].str.contains('K|M')] 

		mapping = dict(K='E3', M='E6', B='E9')
		df_likes_short_num['Likes'] = pd.to_numeric(df_likes_short_num['Likes'].replace(mapping, regex=True))

		df_likes_short_num['likes']=df_likes_short_num['Likes']+3
		df=pd.concat([df_likes_num,df_likes_short_num],ignore_index=False)

		# To extraxt the like from the video. (in videos it contains like "You, Venu Royal Jsp, Harigovind Sathees and 488 others", for this we have to add +3 to 488)
		df_post_video=df[df['Url'].str.contains("/watch/?")]
		df_post_picture=df[~df['Url'].str.contains("/watch/?")]
		# converting short form vies and shares to numeric (ex: 1.2K to 1200)

		df_post_picture['share']=df_post_picture['Share Counts'].str.replace('[^0-9KM.]',' ', flags=re.UNICODE) #(K for thousand and M for million)
		df_share_num=df_post_picture[~df_post_picture['share'].str.contains('K|M')] # pic shares in numbers
		df_share_short_num=df_post_picture[df_post_picture['share'].str.contains('K|M')] 

		mapping = dict(K='E3', M='E6', B='E9')
		df_share_short_num['share'] = pd.to_numeric(df_share_short_num['share'].replace(mapping, regex=True))
		df_picture=pd.concat([df_share_num,df_share_short_num],ignore_index=False)

		df_post_video['views']=df_post_video['Share Counts'].str.replace('[^0-9KM.]',' ', flags=re.UNICODE) #(K for thousand and M for million)
		df_views_num=df_post_video[~df_post_video['views'].str.contains('K|M')] # video views in numbers
		df_views_short_num=df_post_video[df_post_video['views'].str.contains('K|M')] 

		mapping = dict(K='E3', M='E6', B='E9')
		df_views_short_num['views'] = pd.to_numeric(df_views_short_num['views'].replace(mapping, regex=True))
		df_video=pd.concat([df_views_num,df_views_short_num],ignore_index=False)

		df=pd.concat([df_picture,df_video],ignore_index=False)
		df['share'].fillna(0,inplace=True)
		df['views'].fillna(0,inplace=True)

		# converting short form comments to numeric (ex: 1.2K to 1200)
		df['comments']=df['Comments Counts'].str.replace('[^0-9KM.]',' ', flags=re.UNICODE) #(K for thousand and M for million)
		df_comments_num=df[~df['comments'].str.contains('K|M')] # video likes in numbers
		df_comments_short_num=df[df['comments'].str.contains('K|M')] 

		mapping = dict(K='E3', M='E6', B='E9')
		df_comments_short_num['comments'] = pd.to_numeric(df_comments_short_num['Likes'].replace(mapping, regex=True))
		df=pd.concat([df_comments_num,df_comments_short_num],ignore_index=False)

		# converting sLikes, views, share and comments columns to Numeric
		df['likes']=pd.to_numeric(df['likes'], errors='coerce').convert_dtypes()
		df['share']=pd.to_numeric(df['share'], errors='coerce').convert_dtypes()
		df['views']=pd.to_numeric(df['views'], errors='coerce').convert_dtypes()
		df['comments']=pd.to_numeric(df['comments'], errors='coerce').convert_dtypes()

		#Distinguishing post as Videos and Photos
		df_post_video=df[df['Url'].str.contains("/watch/?")] # The post of a Video will contains '/watch/?' and VideoId in the URL.
		df_post_picture=df[~df['Url'].str.contains("/watch/?")] 

		df_summary=df.groupby(by= 'Url')[['likes','comments','views','share']].max().sort_values(['likes'],ascending =False)

		df_summary=df_summary.reset_index()
		df_summary['date']=last_date_of_week
		df_summary=df_summary[['date','Url','likes','comments','views','share']]
		df_summary=df_summary.sort_values('likes')
		df_summary.rename(columns={"Url":"url"},inplace=True)
		df_summary=df_summary.head(5)

		stop_words = set(stopwords.words("english"))
		new_words = ['amp','co', 'http','youtube','http','www','com','href','result ','search','query','result', 'bur', 'channel', 'audience','hai', 'sir', 'youtuber','video','film','please','language','ki','ka','p','se','movie','mera','help','kumar']
		stop_words = stop_words.union(new_words)
		corp = []
		for i in range(0, df.shape[0]):
		    #Remove punctuations
		    text = re.sub('[^a-zA-Z]', ' ', str(df['Comments_Only'].iloc[i])) 
		    text=text.lower()
		    text = text.split()
		    ##Lemmatizing
		    lm = WordNetLemmatizer()   
		    text = [lm.lemmatize(word) for word in text if not word in stop_words] 
		    text = " ".join(text)
		    corp.append(text)   
		df['clean_comments'] = np.array(corp)

		cv=TfidfVectorizer(max_df=0.7, stop_words=stop_words, ngram_range=(1,1), min_df=0.001)
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
		top_df=top_df[['date','word',"freq"]]
		

		sid = SentimentIntensityAnalyzer()
		df['clean_comments']=df['clean_comments'].astype('str')
		df['senti']=df['clean_comments'].apply(sid.polarity_scores)
		df['compsenti'] = df['senti'].apply(lambda x: x['compound'])
		df['possenti'] = df['senti'].apply(lambda x: x['pos'])
		df['negsenti'] = df['senti'].apply(lambda x: x['neg'])
		df['nps']=df['possenti']-df['negsenti']

		df_overall_senti=pd.DataFrame([[last_date_of_week,df['possenti'].mean(),df['negsenti'].mean(),df['compsenti'].mean()]],columns=["date","positive_senti","negative_senti","compo_senti"])

		#most positive profiles by NPS
		df_nps_per_person=((pd.DataFrame(df.groupby(by= 'Person Name')[ 'nps'].mean())).reset_index()).sort_values('nps',ascending=False)
		df_nps_per_person=df_nps_per_person

		#most active profiles 
		df_most_active_person=(pd.DataFrame(df['Person Name'].value_counts().sort_values(ascending=False))).reset_index()
		df_most_active_person.rename(columns={'Person Name':'comments_count','index':'Person Name'},inplace=True)

		#Most active profile with nps
		df_most_active_perfile_and_nps=(pd.merge(df_most_active_person,df_nps_per_person,on='Person Name',how='outer')).sort_values('comments_count',ascending=False)
		df_most_active_perfile_and_nps=df_most_active_perfile_and_nps.head(10)
		df_most_active_perfile_and_nps['date']=last_date_of_week
		df_most_active_perfile_and_nps.rename(columns={"Person Name":"person_name"},inplace=True)
		df_most_active_perfile_and_nps=df_most_active_perfile_and_nps[["date","person_name","comments_count","nps"]]

		#most positive posts
		df_positive_posts=(pd.DataFrame(df.groupby(by= 'Url')[ 'possenti'].mean().sort_values(ascending = False).head(10))).reset_index()
		df_positive_posts["date"]=last_date_of_week
		df_positive_posts.rename(columns={"Url":"url"},inplace=True)
		df_positive_posts=df_positive_posts[["date","url","possenti"]]
		df_positive_posts.rename(columns={'possenti':'positive_sentiment'},inplace=True)

		#most negative posts
		df_negative_posts=(pd.DataFrame(df.groupby(by= 'Url')[ 'negsenti'].mean().sort_values(ascending = False).head(10))).reset_index()
		df_negative_posts["date"]=last_date_of_week
		df_negative_posts.rename(columns={"Url":"url"},inplace=True)
		df_negative_posts=df_negative_posts[["date","url","negsenti"]]
		df_negative_posts.rename(columns={'negsenti':'negative_sentiment'},inplace=True)

		return df_post_count, df_comment_count, df_user_count, df_summary, top_df, df_overall_senti, df_most_active_perfile_and_nps, df_positive_posts, df_negative_posts
		
	def pushdata(df_post_count, df_comment_count, df_user_count, df_summary, top_df, df_overall_senti, df_most_active_perfile_and_nps, df_positive_posts, df_negative_posts):
		engine = create_engine('postgresql://psone_flash_base:psone@kalacitra.in/psone_flash_base')
		
		def trucation(df_post_count, df_comment_count, df_user_count, df_summary, top_df, df_overall_senti, df_most_active_perfile_and_nps, df_positive_posts, df_negative_posts):

			conn = psycopg2.connect(database="psone_flash_base", user='psone_flash_base', password='psone', host='kalacitra.in')  
			cursor = conn.cursor()
			delete1="DELETE FROM fb_post_count"
			cursor.execute(delete1)
			conn.commit()
			delete2="DELETE FROM fb_comments"
			cursor.execute(delete2)
			conn.commit()
			delete3="DELETE FROM fb_user_count"
			cursor.execute(delete3)
			conn.commit()
			delete4="DELETE FROM fb_summary"
			cursor.execute(delete4)
			conn.commit()
			delete5="DELETE FROM fb_freq_words"
			cursor.execute(delete5)
			conn.commit()
			delete6="DELETE FROM fb_sentiment"
			cursor.execute(delete6)
			conn.commit()
			delete7="DELETE FROM fb_most_active_profile"
			cursor.execute(delete7)
			conn.commit()
			delete8="DELETE FROM fb_positive"
			cursor.execute(delete8)
			conn.commit()
			delete9="DELETE FROM fb_negative"
			cursor.execute(delete9)
			conn.commit()
			cursor.close()
			conn.close()
			
			df_post_count.to_sql('fb_post_count',engine,if_exists='append',index=False, method="multi", chunksize=500)
			df_comment_count.to_sql('fb_comments',engine,if_exists='append',index=False, method="multi", chunksize=500)
			df_user_count.to_sql('fb_user_count',engine,if_exists='append',index=False, method="multi", chunksize=500)
			df_summary.to_sql('fb_summary',engine,if_exists='append',index=False, method="multi", chunksize=500)
			top_df.to_sql('fb_freq_words',engine,if_exists='append',index=False, method="multi", chunksize=500)
			df_overall_senti.to_sql('fb_sentiment',engine,if_exists='append',index=False, method="multi", chunksize=500)
			df_most_active_perfile_and_nps.to_sql('fb_most_active_profile',engine,if_exists='append',index=False, method="multi", chunksize=500)
			df_positive_posts.to_sql('fb_positive',engine,if_exists='append',index=False, method="multi", chunksize=500)
			df_negative_posts.to_sql('fb_negative',engine,if_exists='append',index=False, method="multi", chunksize=500)

		def push_to_historical_collection_table(df_post_count, df_comment_count, df_user_count, df_summary, top_df, df_overall_senti, df_most_active_perfile_and_nps, df_positive_posts, df_negative_posts):

			df_post_count.to_sql('fb_post_count_history',engine,if_exists='append',index=False, method="multi", chunksize=500)
			df_comment_count.to_sql('fb_comments_history',engine,if_exists='append',index=False, method="multi", chunksize=500)
			df_user_count.to_sql('fb_user_count_history',engine,if_exists='append',index=False, method="multi", chunksize=500)
			df_summary.to_sql('fb_summary_history',engine,if_exists='append',index=False, method="multi", chunksize=500)
			top_df.to_sql('fb_freq_words_history',engine,if_exists='append',index=False, method="multi", chunksize=500)
			df_overall_senti.to_sql('fb_sentiment_history',engine,if_exists='append',index=False, method="multi", chunksize=500)
			df_most_active_perfile_and_nps.to_sql('fb_most_active_profile_history',engine,if_exists='append',index=False, method="multi", chunksize=500)
			df_positive_posts.to_sql('fb_positive_history',engine,if_exists='append',index=False, method="multi", chunksize=500)
			df_negative_posts.to_sql('fb_negative_history',engine,if_exists='append',index=False, method="multi", chunksize=500)
		
		engine.dispose()

		trucation(df_post_count, df_comment_count, df_user_count, df_summary, top_df, df_overall_senti, df_most_active_perfile_and_nps, df_positive_posts, df_negative_posts)
		#push_to_historical_collection_table(df_post_count, df_comment_count, df_user_count, df_summary, top_df, df_overall_senti, df_most_active_perfile_and_nps, df_positive_posts, df_negative_posts)

	df1, df2 = fetch(uri, mydb)
	df_post_count, df_comment_count, df_user_count, df_summary, top_df, df_overall_senti, df_most_active_perfile_and_nps, df_positive_posts, df_negative_posts = transform(df1, df2)
	#pushdata(df_post_count, df_comment_count, df_user_count, df_summary, top_df, df_overall_senti, df_most_active_perfile_and_nps, df_positive_posts, df_negative_posts)

	return df_post_count, df_comment_count, df_user_count, df_summary, top_df, df_overall_senti, df_most_active_perfile_and_nps, df_positive_posts, df_negative_posts
df_post_count, df_comment_count, df_user_count, df_summary, top_df, df_overall_senti, df_most_active_perfile_and_nps, df_positive_posts, df_negative_posts = FacebookAnalysis(uri,mydb)