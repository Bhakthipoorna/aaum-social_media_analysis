from youtube_analysis_daily import *
from twitter_analysis_daily import *
from overall_nps_entity import *
youtube_tags=['Teaser','Ponni','Chola','Trailer','Ratchasa Maamaney','Alaikadal','Sol','Alaikadal-Glimpse','Sol-Glimpse','Chola-BTS','Promo','Review','Competitor-NaaneVaruvean','Celebrity-Talks']
twitter_tags=['Movie','Ponni','Chola','Trisha','Karthi','Vikram','Aishwarya Rai','Jayam Ravi','Maniratnam','AR Rahman','Review']

def wraperr_overall():
    
    for tag in youtube_tags:
        youtube_analysis_daily("mongodb://localhost:27017","socialmedia",tag)

    for tag in twitter_tags:
        twitter_analysis_daily("mongodb://localhost:27017","socialmedia",tag)

    overall_nps_entity()

    
wraperr_overall()
