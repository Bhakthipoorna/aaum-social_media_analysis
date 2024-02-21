def TwitterAnalysis(uri,mydb):

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
	from nltk.stem.wordnet import WordNetLemmatizer
	from sklearn.feature_extraction.text import CountVectorizer
	from nltk.sentiment.vader import SentimentIntensityAnalyzer
	sid = SentimentIntensityAnalyzer()
	my_date = datetime.date.today() # if date is 01/01/2018
	year, week, day_of_week = my_date.isocalendar()
	last_date_of_week = pd.to_datetime(datetime.date(year, 1, 1) + relativedelta(weeks=+week))
	last_date_of_week=str(last_date_of_week)
	last_date_of_week=last_date_of_week[0:10]

	def fetch_and_transform(uri,mydb):

		uri = "mongodb://localhost:27017"
		client= pym.MongoClient(uri)
		mydb = client['socialmedia']
		page_tweets= mydb['page_tweets']
		hashtag_tweets= mydb['hashtag_tweets_songs']
		df_dumped= pd.json_normalize(hashtag_tweets.find())

		
		df_dumped[["createdAt"]]=df_dumped[["createdAt"]].apply(pd.to_datetime, errors='coerce') 
		df_dumped=df_dumped.sort_values('createdAt')

		#Dropping the tweets with empty Id and empty CreatedAt
		df_dumped=df_dumped[~df_dumped['createdAt'].isnull()]
		df_dumped=df_dumped[~df_dumped['id'].isnull()]

		df_dumped['id']=df_dumped['id'].astype('str')
		df_dumped=df_dumped.sort_values('createdAt')
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

		df['Date']=df['createdAt'].dt.date

		df['text']=df['text'].str.lower()
		df['text'] = df['text'].str.replace('[^A-Za-z0-9]', ' ', flags=re.UNICODE)

		df['text']=df['text'].str.lower()
		df['text'] = df['text'].str.replace('[^A-Za-z0-9]', ' ', flags=re.UNICODE)
		df=df[df['text'].str.contains('teaser|ponniyinselvan|maniratnam|ratnam|ponniyin|selvan|aishwaryarai|ps1|aishwarya rai|lyca|madrastalkies|chola|karthi|vikram|chiyaan|trisha|arrahmah|a r rahman|ar rahman|prakashrai|prakash rai|suriya|jayamravi|ponninadhi|ponni|nadhi|')]
		df['text'] = df['text'].str.replace('[^A-Za-z]', ' ', flags=re.UNICODE)
		df_tweet_count=pd.DataFrame([[last_date_of_week,df.shape[0]]],columns=["Date","TweetCount"])

		df_retweets=df[~df['retweetedStatus.user.screenName'].isnull()]
		df_reply=df[~df['inReplyToScreenName'].isnull()]
		# Creating Dataframe
		df_tweet_dist = pd.DataFrame([[df.shape[0],df_retweets.shape[0],df_reply.shape[0],(df.shape[0]-(df_retweets.shape[0]-df_reply.shape[0]))]],columns =["TotalTweets","Retweets","Replies","DirectTweets"])
		df_tweet_dist['Date']=last_date_of_week

		df_reach=pd.DataFrame([[last_date_of_week,df_retweets.shape[0]/df.shape[0]]],columns=['Date','reach']).reset_index()
		df_reach=df_reach[['Date','reach']]

		df_followers_count=pd.DataFrame(df.groupby(['user.screenName'])['user.followersCount'].max()).reset_index()
		df_top_influencer=(pd.DataFrame(df['retweetedStatus.user.screenName'].value_counts().head(15))).reset_index()
		df_top_influencer.rename(columns={"retweetedStatus.user.screenName":"retweets_count","index":"user.screenName"},inplace=True)
		df_popular=pd.merge(df_top_influencer,df_followers_count,on='user.screenName',how='left')
		df_popular.rename(columns={'user.screenName':'Influencer_screenName'},inplace=True)
		df_popular['Date']=last_date_of_week
		df_popular=df_popular[['Date','Influencer_screenName','retweets_count','user.followersCount']]

		df_more_retweet=(pd.DataFrame(df_retweets['user.screenName'].value_counts().head(15))).reset_index()
		df_more_retweet.columns=['user.screenName','retweeted_count']
		df_more_retweet=pd.merge(df_more_retweet,df_followers_count,on='user.screenName',how='left')
		df_more_retweet['Date']=last_date_of_week
		df_more_retweet=df_more_retweet[['Date','user.screenName','retweeted_count','user.followersCount']]
	

		stop_words = set(stopwords.words("english"))
		new_words = ['amp','co', 'http','youtube','https','www','com','href','result ','search','query','result', 'br', 'channel', 'audience','hai', 'sir', 'youtuber','video','film','please','language','ki','ka','p','se','movie','mera','help','le','de','la','que','un','en','di','il','si','el','che','et','da','pa','na','lo','mi','tu','ko','ne','ya','con','te','je','pc','e','n','l','j','f','k','c','r','du','b','eu','al','qui','sa','ha','rt']
		stop_words = stop_words.union(new_words)
		corp = []
		for i in range(0, len(df)):
		    #Remove punctuations
		    #text = re.sub('[^a-zA-Z]', ' ', df['text'][i])
		    text = re.sub('[^a-zA-Z]', ' ', str(df['text'].iloc[i]))
		    text=text.lower()
		    ##Convert to list from string
		    text = text.split()
		    ##Lemmatizing
		    lm = WordNetLemmatizer() 
		       
		    
		    text = [lm.lemmatize(word) for word in text if not word in stop_words] 
		    text = " ".join(text)
		    corp.append(text)
		df['text']=df['text'].astype('str')
		df["text"]=df["text"].apply(lambda words: ' '.join(word.lower() for word in words.split() if word not in stop_words))

		cv=CountVectorizer(max_df=0.7, stop_words=stop_words, ngram_range=(1,1), min_df=0.001)
		X=cv.fit_transform(corp)
		vector = cv.transform(corp)

		def get_top_n_words(corpus, n=None):
		    vec = cv.fit(corp)
		    bag_of_words = vec.transform(corp)
		    sum_words = bag_of_words.sum(axis=0) 
		    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
		    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
		    return words_freq[:n]

		top_words = get_top_n_words(corp, n=20)
		top_df = pd.DataFrame(top_words)
		top_df.columns=["Word", "Freq"]
		top_df['Date']=last_date_of_week
		top_df=top_df[["Date","Word","Freq"]]

		df['text']=df['text'].astype('str')
		df['senti']=df['text'].apply(sid.polarity_scores)

		df['compsenti'] = df['senti'].apply(lambda x: x['compound'])
		df['possenti'] = df['senti'].apply(lambda x: x['pos'])
		df['negsenti'] = df['senti'].apply(lambda x: x['neg'])
		df['nps'] = df['possenti']-df['negsenti']
		df_nps=pd.DataFrame(df.groupby('Date')['nps'].mean()).reset_index()

		df=df.set_index('createdAt')

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

		df_entity=pd.concat([df_vfx,df_grand,df_director,df_music,df_camera,df_sets,df_history],ignore_index=False)
		df_entity['Date']=last_date_of_week
		df_entity=df_entity[['Date','Entity','Score']]


		return df_tweet_count, df_tweet_dist, top_df, df_nps, df_reach, df_popular, df_more_retweet, df_entity

	def pushdata(df_tweet_count, df_tweet_dist, top_df, df_nps, df_reach, df_popular, df_more_retweet, df_entity):

		#engine = create_engine('postgresql://psone:psone@localhost/psone_flash_base')
		#df_tweet_count.to_sql('tw_tweetcount',engine,if_exists='append',index=False, method="multi")
		#df_tweet_dist.to_sql('tw_tweetdist',engine,if_exists='append',index=False, method="multi")
		#top_df.to_sql('tw_freq_words',engine,if_exists='append',index=False, method="multi")
		#df_nps.to_sql('tw_nps',engine,if_exists='append',index=False, method="multi")
		#df_reach.to_sql('tw_reach',engine,if_exists='append',index=False, method="multi")
		#df_popular.to_sql('tw_popular',engine,if_exists='append',index=False, method="multi")
		#df_more_retweet.to_sql('tw_more_tweets',engine,if_exists='append',index=False, method="multi")
		#df_entity.to_sql('tw_entity',engine,if_exists='append',index=False, method="multi")
		#engine.dispose()

	df_tweet_count, df_tweet_dist, top_df, df_nps, df_reach, df_popular, df_more_retweet, df_entity  = fetch_and_transform(uri,mydb)
	#pushdata(df_tweet_count, df_tweet_dist, top_df, df_nps, df_reach, df_popular, df_more_retweet, df_entity)

	return df_tweet_count, df_tweet_dist, top_df, df_nps, df_reach, df_popular, df_more_retweet, df_entity
df_tweet_count, df_tweet_dist, top_df, df_nps, df_reach, df_popular, df_more_retweet, df_entity  = TwitterAnalysis(uri,mydb)