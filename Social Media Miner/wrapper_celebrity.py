from insta_cast_analysis import *
from twitter_cast_analysis import *
from twitter_media_handles_analysis import *
from celebrity_track import *
def wraperr_celebrity():
    
    insta_cast_analysis("mongodb://localhost:27017","Manual_Data_Push_For_Insta")
    twitter_cast_analysis()
    celebrity_tracker()
    twitter_media_handles_analysis("mongodb://localhost:27017","socialmedia")
   
    
wraperr_celebrity()