import sys
def insta_cast_analysis(uri,mydb):
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

	def fetch(uri,mydb):

		uri = uri#"mongodb://localhost:27017"
		client= pym.MongoClient(uri)
		mydb = client[mydb]#client['Insta']
		insta_comments= mydb['Manual_Data_Push_For_Instagram_Comment_Scapper']
		df_insta= pd.json_normalize(insta_comments.find())
		client.close()

		config_file='insta_config.ini'
		config = ConfigParser()
		config.read(config_file)
		cast_names=config['CAST_NAME']['cast_names']

		df_lyca_production_lookup=pd.read_excel("Insta_PS1_lookup_lyca_productions.xlsx",engine='openpyxl')
		df_madras_talkies_lookup=pd.read_excel("Insta_PS1_lookup_madras_talkies.xlsx",engine='openpyxl')
		df_jayamravi_lookup=pd.read_excel("Insta_PS1_lookup_jayamravi.xlsx",engine='openpyxl')
		df_aishwaryarai_lookup=pd.read_excel("Insta_PS1_lookup_aishwaryarai.xlsx",engine='openpyxl')
		df_trisha_lookup=pd.read_excel("Insta_PS1_lookup_trisha.xlsx",engine='openpyxl')
		df_karthi_lookup=pd.read_excel("Insta_PS1_lookup_karthi.xlsx",engine='openpyxl')
		df_arrahman_lookup=pd.read_excel("Insta_PS1_lookup_arrahman.xlsx",engine='openpyxl')
		df_manirathnam_lookup=pd.read_excel("Insta_PS1_lookup_maniratnam.xlsx",engine='openpyxl')
		df_prakashraj_lookup=pd.read_excel("Insta_PS1_lookup_prakashraj.xlsx",engine='openpyxl')
		df_vikramprabhu_lookup=pd.read_excel("Insta_PS1_lookup_vikramprabhu.xlsx",engine='openpyxl')
		df_vikram_lookup=pd.read_excel("Insta_PS1_lookup_vikram.xlsx",engine='openpyxl')

		df_lookup=pd.concat([df_lyca_production_lookup,df_madras_talkies_lookup,df_jayamravi_lookup,df_aishwaryarai_lookup,df_trisha_lookup,df_karthi_lookup,df_arrahman_lookup,df_manirathnam_lookup,df_prakashraj_lookup,df_vikramprabhu_lookup, df_vikram_lookup],ignore_index=False)
		return df_insta, cast_names, df_lookup

	def transform(df_insta, cast_names, df_lookup):

		df_insta['Follower Count']=df_insta['Follower Count'].fillna(0)
		df_insta['Follower Count']=df_insta['Follower Count'].str[:-9]
		df_insta['Follower Count']=df_insta['Follower Count'].str.strip()
		df_insta['Follower Count']=df_insta['Follower Count'].fillna(0)
		df_followers_num=df_insta[~df_insta['Follower Count'].str.contains('K|M|B',na=False)]
		df_followers_short_num=df_insta[df_insta['Follower Count'].str.contains('K|M|B',na=False)]
		mapping = dict(K='E3', M='E6', B='E9')
		df_followers_short_num['Follower Count'] = pd.to_numeric(df_followers_short_num['Follower Count'].replace(mapping, regex=True))
		df_insta=pd.concat([df_followers_num,df_followers_short_num],ignore_index=False)
		df_insta['Follower Count']=pd.to_numeric(df_insta['Follower Count'], errors='coerce').convert_dtypes()
		df_followers_count=pd.DataFrame(df_insta.groupby('Name of the Page')['Follower Count'].max()).reset_index()
		

		df_lookup['Image_Url']=df_lookup['Image_Url'].str.strip()
		df_lookup['Image_Url']=df_lookup['Image_Url'].str.lstrip()
		df_lookup['Image_Url']=df_lookup['Image_Url'].str.rstrip()
		df_lookup['Image_Url']=df_lookup['Image_Url'].astype('str')
		df_lookup_image=df_lookup[~df_lookup['Image_Url'].str.contains('reel/')]

		df_lookup_video=df_lookup[df_lookup['Image_Url'].str.contains('reel/')] # In the lookup page video links contains 'reels/' in the link but the in the collection video links does not contain 'reel/'
		df_lookup_video['Image_Url']=df_lookup_video['Image_Url'].str.replace("reel/","p/") #Ex: 'https://www.instagram.com/p/CfgEMTlDMTo/' in the collection and 'https://www.instagram.com/reels/CfgEMTlDMTo/' in the lookup page

		df_lookup_image['Image_Url']=df_lookup_image['Image_Url'].str[0:-28] # Url in lookup page conatins '?utm_source=ig_web_copy_link' at the end, we are removing this.
		df_lookup_video['Image_Url']=df_lookup_video['Image_Url'].str[0:-28]

		df_picture=df_insta[df_insta['Image_Url'].isin(df_lookup_image['Image_Url'])]
		df_video=df_insta[df_insta['Image_Url'].isin(df_lookup_video['Image_Url'])]

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


		df_cast_picture_data=df_picture[df_picture['Name of the Page'].str.contains(cast_names)]
		df_cast_video_data=df_video[df_video['Name of the Page'].str.contains(cast_names)]
		df_cast_data=pd.concat([df_cast_picture_data,df_cast_video_data],ignore_index=False)

		df['Date of Post']=pd.to_datetime(df['Date of Post'])


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
		                   ngram_range=(1,1), 
		                   min_df=0.001)
		X=cv.fit_transform(corp)
		vector = cv.transform(corp)


		df['clean_comments']=df['clean_comments'].astype('str')
		df['senti']=df['clean_comments'].apply(sid.polarity_scores)

		df['compsenti'] = df['senti'].apply(lambda x: x['compound'])
		df['possenti'] = df['senti'].apply(lambda x: x['pos'])
		df['negsenti'] = df['senti'].apply(lambda x: x['neg'])
		df['nps']=df['possenti']-df['negsenti']
		df['date']=df['Date of Post'].dt.date
		df_nps_over_time=pd.DataFrame(df.groupby('date')['nps'].mean()).reset_index()


		#most positive profiles by NPS
		df_nps_per_person=(pd.DataFrame(df.groupby(by= 'Commented_Person_name')[ 'nps'].mean())).reset_index()
		df_nps_per_person.columns=["person_name","nps"] 
		df_nps_per_person=df_nps_per_person[["person_name","nps"]]
		df_nps_per_person=df_nps_per_person.sort_values('nps',ascending=False).head(10)

		df_castwise=pd.DataFrame(df_cast_data.groupby(['Name of the Page','Image_Url'])['Likes','Views','Comments Counts'].max()).reset_index()

		df_castwise['Likes']=pd.to_numeric(df_castwise['Likes'], errors='coerce').convert_dtypes()
		df_castwise['Comments Counts']=pd.to_numeric(df_castwise['Comments Counts'], errors='coerce').convert_dtypes()
		df_castwise['Views']=pd.to_numeric(df_castwise['Views'], errors='coerce').convert_dtypes()

		df_cast_image=pd.DataFrame(df_cast_picture_data.groupby(['Name of the Page'])['Image_Url'].nunique()).reset_index()
		df_cast_image.rename(columns={"Image_Url":"image_posts"},inplace=True)
		df_cast_video=pd.DataFrame(df_cast_video_data.groupby(['Name of the Page'])['Image_Url'].nunique()).reset_index()
		df_cast_video.rename(columns={"Image_Url":"video_posts"},inplace=True)
		df_cast_post_count=pd.DataFrame(df_cast_data.groupby(['Name of the Page'])['Image_Url'].nunique()).reset_index()
		df_cast_post_count.rename(columns={"Image_Url":"Related_posts"},inplace=True)
		df_cast_post_counts=pd.merge(df_cast_post_count,df_cast_image,on='Name of the Page',how='outer').merge(df_cast_video,on='Name of the Page',how='outer')
		df_castwise_report=pd.DataFrame(df_castwise.groupby(['Name of the Page'])[['Likes','Views','Comments Counts']].sum()).reset_index()

		df_castwise_summary=pd.merge(df_castwise_report,df_cast_post_counts,on='Name of the Page',how='outer')
		df_castwise_summary=df_castwise_summary[["Name of the Page","Related_posts","image_posts","video_posts","Likes","Views","Comments Counts"]]
		df_castwise_summary.fillna(0,inplace=True)
		df_castwise_summary=pd.merge(df_followers_count,df_castwise_summary,on='Name of the Page',how='left')

		df_castwise_summary.rename(columns={"Name of the Page":"cast_name","Related_posts":"related_posts","Likes":"likes","Views":"views","Comments Counts":"comments"},inplace=True)
		cast_modified_names=['Maniratnam','Vikram','Jayam Ravi','Karthi','Aishwarya Rai Bachchan','Trisha Krishnan','Vikram Prabhu','Ravi Varman','A Sreekar Prasad','A R Rahman','B Jeyamohan','Prakash Raj','Lyca Productions','Madras Talkies']
		df_castwise_summary['cast_name'].replace({'maniratnam.official':'Maniratnam','jayamravi_official':'Jayam Ravi','karthi_offl':'Karthi','aishwaryaraibachchan_arb':'Aishwarya Rai Bachchan','trishakrishnan':'Trisha Krishnan','arrahman':'A R Rahman','joinprakashraj':'Prakash Raj','lyca_productions':'Lyca Productions','madrastalkies':'Madras Talkies','iamvikramprabhu':'Vikram Prabhu','the_real_chiyaan':'Vikram'},inplace=True)
		cast_modified_names=pd.DataFrame(cast_modified_names)
		cast_modified_names.columns=['cast_name']
		df_castwise_summary=pd.merge(cast_modified_names,df_castwise_summary,on='cast_name',how='left')
		df_castwise_summary.rename(columns={'Follower Count':'followers_count'},inplace=True)
		df_castwise_summary.fillna(0,inplace=True)

		return df_castwise_summary

	def pushdata(df_castwise_summary):

		conn = psycopg2.connect(database="psone_flash_base", user='psone_flash_base', password='psone', host='kalacitra.in')  
		cursor = conn.cursor()
		delete1="DELETE FROM insta_cast_analysis"
		cursor.execute(delete1)
		conn.commit()
		cursor.close()
		conn.close()

		engine = create_engine('postgresql://psone_flash_base:psone@kalacitra.in/psone_flash_base')
		df_castwise_summary.to_sql('insta_cast_analysis',engine,if_exists='append',index=False, method="multi", chunksize=500)
		engine.dispose()

	df_insta, cast_names, df_lookup = fetch(uri,mydb)
	df_castwise_summary = transform(df_insta, cast_names, df_lookup)
	pushdata(df_castwise_summary)
