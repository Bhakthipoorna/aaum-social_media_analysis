
import sys
def youtube_analysis_weekly(uri,mydb,tag):

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
    import psycopg2
    from sqlalchemy import create_engine
    from datetime import datetime,timedelta
    import pytz
    IST = pytz.timezone('Asia/Kolkata')
    Today=pd.to_datetime(datetime.strftime(datetime.now(IST), '%Y-%m-%d'))
    yesterday=datetime.strftime(datetime.now(IST) - timedelta(1), '%Y-%m-%d')
    prior_7days=datetime.strftime(datetime.now(IST) - timedelta(7), '%Y-%m-%d')

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


        df[['updated_at']]=df[["updated_at"]].apply(pd.to_datetime, errors='coerce')
        df=df[df['updated_at']<str(Today)]
        df=df[df['updated_at']>=prior_7days]
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
        df_likes_views_comments['date']=yesterday
        df_likes_views_comments['date']=pd.to_datetime(df_likes_views_comments['date'])
        df_likes_views_comments['tag']=tag
        df_likes=df_likes_views_comments[['date','tag','language','like_count']]
        df_views=df_likes_views_comments[['date','tag','language','views_count']]
        df_comments=df_likes_views_comments[['date','tag','language','comment_count']]

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
        if len(df['comments_text'])>10:
            top_words = get_top_n_words(corp, n=20)
            top_df = pd.DataFrame(top_words)
            top_df.columns=["word", "freq"]
            top_df['date']=yesterday
            top_df['tag']=tag
            drop_words=['please','others','ponniyin','selvan','ponniyin selvan']
            top_df=top_df[~top_df['word'].isin(drop_words)]
            top_df=top_df.head(5)
            top_df=top_df[['date','tag','word','freq']]
        else :
            top_df=pd.DataFrame()

        df['comments_text']=df['comments_text'].astype('str')
        df['senti']=df['comments_text'].apply(sid.polarity_scores)

        df['compsenti'] = df['senti'].apply(lambda x: x['compound'])
        df['possenti'] = df['senti'].apply(lambda x: x['pos'])
        df['negsenti'] = df['senti'].apply(lambda x: x['neg'])

        df=df.set_index('updated_at')
        df['nps']= df['possenti']-df['negsenti']
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
        df_positive['date']=yesterday
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
        df_negative['date']=yesterday
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
        df_positive_profile=df_positive_profile[df_positive_profile['nps']>0]
        df_positive_profile=df_positive_profile.head(10)
        df_positive_profile['date']=yesterday
        df_positive_profile['date']=pd.to_datetime(df_positive_profile['date'])
        df_positive_profile['tag']=tag
        df_positive_profile=df_positive_profile[['date','tag','person_name','comments_count','nps']]

        df_negative_profile=df_nps_per_person.sort_values(['comments_count'],ascending=False)
        df_negative_profile=df_negative_profile[df_negative_profile['nps']<0]
        df_negative_profile=df_negative_profile.head(10)
        df_negative_profile['date']=yesterday
        df_negative_profile['date']=pd.to_datetime(df_negative_profile['date'])
        df_negative_profile['tag']=tag
        df_negative_profile=df_negative_profile[['date','tag','person_name','comments_count','nps']]

        return df_views, df_likes, df_comments, top_df, df_positive_negative, df_positive_profile, df_negative_profile


    def pushdata(df_views, df_likes, df_comments, top_df, df_positive_negative, df_positive_profile, df_negative_profile):
        engine = create_engine('postgresql://psone_flash_base:psone@kalacitra.in/psone_flash_base')
        df_views.to_sql('yt_views',engine,if_exists='append',index=False, method="multi")
        df_likes.to_sql('yt_likes',engine,if_exists='append',index=False, method="multi")
        df_comments.to_sql('yt_comments',engine,if_exists='append',index=False, method="multi")
        top_df.to_sql('yt_freq_words',engine,if_exists='append',index=False, method="multi")
        df_positive_negative.to_sql('yt_positive_negative',engine,if_exists='append',index=False, method="multi")
        df_positive_profile.to_sql('yt_positive_profile',engine,if_exists='append',index=False, method="multi")
        df_negative_profile.to_sql('yt_negative_profile',engine,if_exists='append',index=False, method="multi")
        engine.dispose()

    df = fetch(uri,mydb,tag)
    df_views, df_likes, df_comments, top_df, df_positive_negative, df_positive_profile, df_negative_profile = transform(df)
    pushdata(df_views, df_likes, df_comments, top_df, df_positive_negative, df_positive_profile, df_negative_profile)

uri = sys.argv[1]
mydb = sys.argv[2]
tags = sys.argv[3]
tags = tags.split(',')
for tag in tags:
    youtube_analysis_weekly(uri,mydb,tag)

