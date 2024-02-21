
def twitter_competitor_analysis_daily(uri,mydb, tag):
    import pandas as pd
    import numpy as np
    import re
    import pymongo as pym
    import nltk
    import psycopg2
    from sqlalchemy import create_engine
    from nltk.corpus import stopwords
    from dateutil.relativedelta import relativedelta
    from configparser import ConfigParser
    from nltk.stem.wordnet import WordNetLemmatizer
    from sklearn.feature_extraction.text import CountVectorizer
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    from datetime import datetime,timedelta
    import pytz
    IST = pytz.timezone('Asia/Kolkata')
    Today=pd.to_datetime(datetime.strftime(datetime.now(IST), '%Y-%m-%d'))
    yesterday=datetime.strftime(datetime.now(IST) - timedelta(1), '%Y-%m-%d')

    def fetch(uri, mydb):
        uri = "mongodb://localhost:27017"
        client= pym.MongoClient(uri)
        mydb = client['socialmedia']
        hashtag_tweets= mydb['competitor_hashtag_tweets']
        df_dumped= pd.json_normalize(hashtag_tweets.find())
        client.close()

        return df_dumped

    def transform(df_dumped):
        df_dumped[["createdAt"]]=df_dumped[["createdAt"]].apply(pd.to_datetime, errors='coerce') 
        df_dumped=df_dumped.sort_values('createdAt')
        df_dumped=df_dumped[~df_dumped['createdAt'].isnull()]
        df_dumped=df_dumped[~df_dumped['id'].isnull()]
        df_dumped['id']=df_dumped['id'].astype('str')
        df_dumped=df_dumped.sort_values('createdAt')
        df_dumped=df_dumped[df_dumped['createdAt']>=yesterday]
        df_dumped=df_dumped[df_dumped['createdAt']<Today]
        df=df_dumped.drop_duplicates(['id'],keep='last') 

        df['text']=df['text'].str.lower()
        df['text'] = df['text'].str.replace('[^A-Za-z]', ' ', flags=re.UNICODE)

        if tag=='Vikram Vedha':
            df=df[df['text'].str.contains('vikram|vedha|hrithik|saif|radhika apte|radhikaapte|yogitabihani')]
        elif tag=='Naane Varuvean':
            df=df[df['text'].str.contains('naane|varuvean|varuven|dhanush|selvaraghavan|vcreations|veerasoora|veera soora')]
        elif tag=='Brahmastra':
            df=df[df['text'].str.contains('brahmastra|shiva|amitabh|ranbir|kapoor|alia|mouni roy|part one|karan|johar')]


        if len(df)<1:

            df_daily_tweet_count=pd.DataFrame([[yesterday,0,tag]],columns=['date','total_tweets','tag'])
            df_daily_tweetdist=pd.DataFrame([[yesterday,tag,0,0,0,0]],columns=['date','tag','total_tweets','retweets','replies','direct_tweets'])
            df_daily_tweetdist_retweet=pd.DataFrame([[yesterday,0,tag]],columns=['date','retweets','tag'])
            df_daily_tweetdist_replies=pd.DataFrame([[yesterday,0,tag]],columns=['date','replies','tag'])
            df_daily_tweetdist_direct_tweets=pd.DataFrame([[yesterday,0,tag]],columns=['date','direct_tweets','tag'])
            df_nps=pd.DataFrame([[yesterday,0,tag]],columns=['date','nps','tag'])
            df_sentiment=pd.DataFrame(['Positive Sentiment','Negative Sentiment'])
            df_sentiment['date']=yesterday
            df_sentiment['tag']=tag
            df_sentiment['score']=0
            df_reach=pd.DataFrame([[yesterday,0,tag]],columns=['date','reach','tag'])
            df_entity=pd.DataFrame(['vfx','history','grand','sets','director','camera','music'])
            df_entity['date']=yesterday
            df_entity['tag']=tag
            df_entity.columns=['entity','date','tag']
            df_entity['percentage']=0

        elif len(df)>=1:
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
            df['date']=df['createdAt'].dt.date

            df['tweet_date']=df['createdAt'].dt.date

            df_retweets=df[~df['retweetedStatus.user.screenName'].isnull()]
            df_reply=df[~df['inReplyToScreenName'].isnull()]

            df_daily_tweet_count=pd.DataFrame(df['tweet_date'].value_counts()).reset_index()
            df_daily_tweet_count.columns=['date','total_tweets']
            df_daily_tweet_count=df_daily_tweet_count.sort_values('date')
            df_daily_tweet_count['date']=df_daily_tweet_count['date'].astype('object')
            df_daily_tweet_count['tag']=tag

            df_daily_retweet=pd.DataFrame(df_retweets.groupby('tweet_date')['createdAt'].count()).reset_index()
            df_daily_retweet.columns=['date','retweets']
            df_daily_retweet['date']=df_daily_retweet['date'].astype('object')
            df_daily_replies=pd.DataFrame(df_reply.groupby('tweet_date')['createdAt'].count()).reset_index()
            df_daily_replies.columns=['date','replies']
            df_daily_replies['date']=df_daily_retweet['date'].astype('object')

            df_retweets_reply=pd.merge(df_daily_retweet,df_daily_replies,on='date',how='outer')
            df_daily_tweetdist=pd.merge(df_retweets_reply,df_daily_tweet_count,on='date',how='outer')
            df_daily_tweetdist.fillna(0,inplace=True)
            df_daily_tweetdist['direct_tweets']=abs(df_daily_tweetdist['total_tweets']-(df_daily_tweetdist['retweets']+df_daily_tweetdist['replies']))
            df_daily_tweetdist['tag']=tag
            df_daily_tweetdist_retweet=df_daily_tweetdist[['date','tag','retweets']]
            df_daily_tweetdist_replies=df_daily_tweetdist[['date','tag','replies']]
            df_daily_tweetdist_direct_tweets=df_daily_tweetdist[['date','tag','direct_tweets']]

            df_direct_tweets=df_daily_tweetdist[['date','direct_tweets']]
            df_favourite_count=pd.DataFrame(df.groupby('date')['user.favouritesCount'].sum()).reset_index()

            df_reach1=(pd.merge(df_direct_tweets,df_favourite_count,on='date',how='outer')).merge(df_daily_retweet,on='date',how='outer')
            df_reach=df_reach1[df_reach1['direct_tweets']>0]
            if len(df_reach)>=1:
                df_reach['reach']=(df_reach['retweets']+df_reach['user.favouritesCount'])
                df_reach['reach']=df_reach['reach']/df_reach['direct_tweets']
                df_reach['reach'] = df_reach['reach'].astype(int, errors='ignore')
                df_reach['date']=df_reach['date'].apply(pd.to_datetime, errors='coerce')
                df_reach=df_reach.fillna(0)
                df_reach['tag']=tag
                df_reach=df_reach[['date','tag','reach']]
            else:
                df_reach1['reach']=(df_reach1['retweets']+df_reach1['user.favouritesCount'])
                df_reach=df_reach1
                df_reach=df_reach[['date','reach']]
                df_reach['tag']=tag

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

            #Removing stopwords from the text column
            df['text']=df['text'].astype('str')
            df["text"]=df["text"].apply(lambda words: ' '.join(word.lower() for word in words.split() if word not in stop_words))

            cv=CountVectorizer(max_df=0.7, stop_words=stop_words, ngram_range=(1,2), min_df=0.001)
            X=cv.fit_transform(corp)
            vector = cv.transform(corp)

            df['text']=df['text'].astype('str')
            df['senti']=df['text'].apply(sid.polarity_scores)

            df['compsenti'] = df['senti'].apply(lambda x: x['compound'])
            df['possenti'] = df['senti'].apply(lambda x: x['pos'])
            df['negsenti'] = df['senti'].apply(lambda x: x['neg'])
            df[['negsenti','possenti']]=df[['negsenti','possenti']]*100
            df['nps'] = df['possenti']-df['negsenti']

            df_nps=pd.DataFrame(df.groupby('date')['nps','possenti','negsenti'].mean()).reset_index()
            df_nps['tag']=tag

            nps=df_nps[['date','tag','nps']]
            nps['metric']='nps'
            nps.rename(columns={'nps':'score'},inplace=True)
            possenti=df_nps[['date','tag','possenti']]
            possenti['metric']='possenti'
            possenti.rename(columns={'possenti':'score'},inplace=True)
            negsenti=df_nps[['date','tag','negsenti']]
            negsenti['metric']='negsenti'
            negsenti.rename(columns={'negsenti':'score'},inplace=True)
            df_sentiment=pd.concat([nps,possenti,negsenti],ignore_index=False)
            df_sentiment=df_sentiment.fillna(0)
            df_sentiment=df_sentiment[['date','tag','metric','score']]
            df_sentiment['metric'].replace({'possenti':'positive','negsenti':'negative'},inplace=True)

            df=df.set_index('createdAt') #df_more_retweets

            df['vfx'] = df['text'].str.contains('vfx',case = False)
            vfx=pd.DataFrame(df['vfx'].resample('1d').sum().reset_index())
            vfx.rename(columns={'createdAt':'date'},inplace=True)
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

            df['grand'] = df['text'].str.contains('grand',case = False)
            grand=pd.DataFrame(df['grand'].resample('1d').sum().reset_index())
            grand.rename(columns={'createdAt':'date'},inplace=True)
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

            df['director'] = df['text'].str.contains('director',case = False)
            director=pd.DataFrame(df['director'].resample('1d').sum().reset_index())
            director.rename(columns={'createdAt':'date'},inplace=True)
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

            df['music'] = df['text'].str.contains('music',case = False)
            music=pd.DataFrame(df['music'].resample('1d').sum().reset_index())
            music.rename(columns={'createdAt':'date'},inplace=True)
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

            df['camera'] = df['text'].str.contains('camera',case = False)
            camera=pd.DataFrame(df['camera'].resample('1d').sum().reset_index())
            camera.rename(columns={'createdAt':'date'},inplace=True)
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

            df['sets'] = df['text'].str.contains('sets',case = False)
            sets=pd.DataFrame(df['sets'].resample('1d').sum().reset_index())
            sets.rename(columns={'createdAt':'date'},inplace=True)
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

            df['history'] = df['text'].str.contains('history',case = False)
            history=pd.DataFrame(df['history'].resample('1d').sum().reset_index())
            history.rename(columns={'createdAt':'date'},inplace=True)
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

        return df_daily_tweet_count,df_daily_tweetdist, df_daily_tweetdist_retweet,df_daily_tweetdist_replies, df_daily_tweetdist_direct_tweets, df_nps, df_sentiment, df_reach, df_entity

    def pushdata(df_daily_tweet_count, df_daily_tweetdist,df_daily_tweetdist_retweet,df_daily_tweetdist_replies, df_daily_tweetdist_direct_tweets, df_nps, df_sentiment, df_reach, df_entity):

            engine = create_engine('postgresql://psone_flash_base:psone@kalacitra.in/psone_flash_base')
            df_daily_tweet_count.to_sql('tw_daily_tweetcount',engine,if_exists='append',index=False, method="multi")
            df_daily_tweetdist.to_sql('tw_daily_tweetdist',engine,if_exists='append',index=False, method="multi")
            df_daily_tweetdist_retweet.to_sql('tw_daily_tweetdist_retweet',engine,if_exists='append',index=False, method="multi")
            df_daily_tweetdist_replies.to_sql('tw_daily_tweetdist_replies',engine,if_exists='append',index=False, method="multi")
            df_daily_tweetdist_direct_tweets.to_sql('tw_daily_tweetdist_direct_tweets',engine,if_exists='append',index=False, method="multi")
            df_nps.to_sql('tw_nps',engine,if_exists='append',index=False, method="multi")
            df_sentiment.to_sql('tw_sentiment',engine,if_exists='append',index=False, method="multi")
            df_reach.to_sql('tw_reach',engine,if_exists='append',index=False, method="multi")
            df_entity.to_sql('tw_entity',engine,if_exists='append',index=False, method="multi")
            engine.dispose()

    df_dumped = fetch(uri,mydb)
    df_daily_tweet_count, df_daily_tweetdist, df_daily_tweetdist_retweet,df_daily_tweetdist_replies, df_daily_tweetdist_direct_tweets, df_nps, df_sentiment, df_reach, df_entity = transform(df_dumped)
    #pushdata(df_daily_tweet_count,df_daily_tweetdist, df_daily_tweetdist_retweet,df_daily_tweetdist_replies, df_daily_tweetdist_direct_tweets, df_nps, df_sentiment, df_reach, df_entity)

uri = sys.argv[1]
mydb = sys.argv[2]
tags = sys.argv[3]
tags = tags.split(',')
for tag in tags:
    twitter_competitor_analysis_daily(uri,mydb,tag)
