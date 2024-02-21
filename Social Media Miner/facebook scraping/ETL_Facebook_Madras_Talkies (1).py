#!/usr/bin/env python
# coding: utf-8

# In[34]:


import sys
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select,WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
import pandas as pd
import numpy as np
import datetime
import json
from datetime import timedelta
import time
import os
from dateutil.tz import gettz
import warnings
warnings.filterwarnings("ignore")
import requests
from difflib import SequenceMatcher
from selenium.common.exceptions import NoAlertPresentException
import re
import string
option = Options()
path=ChromeDriverManager().install()
option.add_argument("--disable-infobars")
option.add_argument("start-maximized")
option.add_argument("--incognito")
# option.add_argument("--headless")
option.add_argument("--window-size=1920,1080")
#option.add_argument("--window-size=1382,754")
# Pass the argument 1 to allow and 2 to block
option.add_experimental_option("prefs", { 
    "profile.default_content_setting_values.notifications": 2
})


# In[35]:


def login_credential():
    faceboo_url="https://www.facebook.com/"
    # fb_username="7010413012"
    # fb_password="aaum123"

    # fb_username="sundar_sp_0002"
    # fb_password="aaum@1234"
    fb_username="poyyinselvan1947@gmail.com"
    fb_password="Ponniyin@567"

    page_url="https://www.facebook.com/madrastalkiesofficial"

    file_name="Facebook_Full_Data_Till_Sep_01_2022.csv"
    return faceboo_url,fb_username,fb_password,page_url,file_name

# faceboo_url,fb_username,fb_password,page_url,file_name=login_credential()[0],login_credential()[1],login_credential()[2],login_credential()[3],login_credential()[4]
# print(faceboo_url)
# print(fb_username)
# print(fb_password)
# print(page_url)
# print(file_name)


# In[38]:


def facebook_2():
    
    login_credential()
    
    facebook_url,fb_username,fb_password,page_url,file_name=login_credential()[0],login_credential()[1],login_credential()[2],login_credential()[3],login_credential()[4]
    print(facebook_url)
    print(fb_username)
    print(fb_password)
    print(page_url)
    print(file_name)
    
    def previous_data_load(file_name):
        all_data=pd.read_csv(file_name,index_col=0)
        all_data.drop('_id',axis=1,inplace=True)
        list_of_all_old_url=list(all_data['Url'].unique())
        print("Length of Old Url Unique Counts :",len(list_of_all_old_url))
        return list_of_all_old_url

    previous_data=previous_data_load(file_name)
    print(previous_data)
    
    def facebook_data_extraction():

        datafr=pd.DataFrame(columns=['Name of the Page','Url','Person Name','Person Url','Likes Counts','Comments Counts','Share Counts','Likes_Comments_Shares_Counts','Comments'])

    #     driver = webdriver.Chrome(options=option, executable_path="C:/ChromeDriver_Folder/chromedriver.exe")
        driver = webdriver.Chrome(options=option, executable_path=path)
        driver.get(facebook_url)

    #     print("1",driver.current_url)
        time.sleep(2)
        ele_username=driver.find_element(By.ID,"email")
        ele_username.send_keys(fb_username)
        time.sleep(1)
        ele_password=driver.find_element(By.ID,"pass")
        ele_password.send_keys(fb_password)
        time.sleep(1)
        ele_login_button=driver.find_element(By.NAME,"login")
        ele_login_button.click()

    #     print("1",driver.current_url)
        time.sleep(5)
        driver.get(page_url)
    #     print("1",driver.current_url)

        time.sleep(5)
        try:
            for loop in range(1,10):
                time.sleep(2)
                driver.find_element_by_tag_name('body').send_keys(Keys.END)
        except:
            pass
        time.sleep(5)
        list_of_get_url=driver.find_elements(By.XPATH,"//div[@class='om3e55n1']//a")
        print("URL---------->",len(list_of_get_url))
        time.sleep(2)
        get_post_url=[]
        for url_post in list_of_get_url:
            get_post_url.append(url_post.get_attribute("href"))

    #     print("=============>",get_post_url)

        time.sleep(2)
        phot_url=[]
        video_url=[]
        youtube_url=[]
        for ii in get_post_url:
            if('photos' in ii):
                phot_url.append(ii)
            elif('Playlist' in ii):
                youtube_url.append(ii)
            elif("videos" in ii):
                video_url.append(ii)

        all_url=phot_url+video_url
        print("PHOTO AND VIDEO URL------------------>",all_url)

        for mk in all_url:
            print(mk)
            print("__________________________________________________________________")

        print("PHOTO",len(phot_url))
        print("VIDEOS",len(video_url))
        print("YOUTUBE",len(youtube_url))
        print("PHOTO AND VIDEOS",len(all_url))
        time.sleep(3)

        for each_post_url in all_url:
            time.sleep(3)
            try:
                driver.get(each_post_url)
            except:
                pass

            time.sleep(3)
            current_url=driver.current_url
            print("Current URL------------------->",current_url)
            if (current_url not in previous_data):
    #             current_post_url.append(old_one)
                print("Current Available New URL",current_url)
                print("Current New URL---------->",current_url)

    #             time.sleep(5)
    #             try:
    #                 get_page_name=driver.find_element(By.XPATH,"//div[@class='i0rxk2l3']//h2").text
    #     #             print("Page Name: ",get_page_name)
    #             except:
    #                 pass

                # For Videos Clicking Relevant Button
                time.sleep(2)
                try:
                    ele_click_most_relevent_button=driver.find_element(By.XPATH,"//div[@class='']//span[text()='Most relevant']")
                    ele_click_most_relevent_button.click()
                except:
                    pass

                time.sleep(2)
                try:
                    ele_click_all_comments=driver.find_element(By.XPATH,"//span[text()='All comments']")
                    ele_click_all_comments.click()
                except:
                    pass
                time.sleep(2)
                try:
                    click_to_pause_video=driver.find_element(By.XPATH,"(//div[@class='z6erz7xo bdao358l on4d8346 s8sjc6am myo4itp8 ekq1a7f9 fsf7x5fv'])[1]")
                    click_to_pause_video.click()
                except:
                    pass
                time.sleep(3)
                for i in range(0,10):
                    try:
                        ele_click_extra_comments_20=driver.find_elements(By.XPATH,"//*[contains(@id,'mount_0_0_')]//span[@class='alzwoclg lxowtz8q aeinzg81']")
                        print("Length of View More Comments Clicked",len(ele_click_extra_comments_20))
                        if(len(ele_click_extra_comments_20)>0):
                            ele_click_extra_comments=driver.find_element(By.XPATH,"//*[contains(@id,'mount_0_0_')]//span[@class='alzwoclg lxowtz8q aeinzg81']")
                            driver.execute_script("arguments[0].click();", ele_click_extra_comments)
                            print("Inside For Loop View More Comments Clicked")
                        else:
                            pass
                    except:
                        pass

                time.sleep(5)
                try:
                    time.sleep(2)
                    ele_click_extra_comments=driver.find_element(By.XPATH,"//*[contains(@id,'mount_0_0_')]//span[@class='alzwoclg lxowtz8q aeinzg81']")
                    driver.execute_script("arguments[0].click();", ele_click_extra_comments)
                except:
                    pass
                try:
                    time.sleep(2)
                    ele_click_extra_comments=driver.find_element(By.XPATH,"//*[contains(@id,'mount_0_0_')]//span[@class='alzwoclg lxowtz8q aeinzg81']")
                    driver.execute_script("arguments[0].click();", ele_click_extra_comments)
                except:
                    pass

                try:
                    get_videos_count_likes_view=driver.find_elements(By.XPATH,"(//div[@class='hf30pyar lq84ybu9 pdfqpcpb siu44isn o7bt71qk alzwoclg rtxb060y i85zmo3j'])[1]")
        #             print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^",len(get_videos_count_likes_view))
                except:
                    pass

                if(len(get_videos_count_likes_view)>0):
                    try:
                        get_page_name=driver.find_element(By.XPATH,"//*[contains(@id,'watch_feed')]/div/div[1]/div/div/div[2]/div[1]/div[1]/div[2]/div/div[1]/span/h2").text
            #             print("Page Name: ",get_page_name)
                    except:
                        pass
        #             print("*****************************INSIDE IF**************************************")
                    time.sleep(3)
                    try:
                        get_videos_count_likes_view=driver.find_element(By.XPATH,"(//div[@class='hf30pyar lq84ybu9 pdfqpcpb siu44isn o7bt71qk alzwoclg rtxb060y i85zmo3j'])[1]").text
                        print(get_videos_count_likes_view)
                        get_videos_count_likes_view=get_videos_count_likes_view.split("\n")[::2]
                        print(get_videos_count_likes_view)
                        get_videos_count_likes_view=",".join(get_videos_count_likes_view)
                        print(get_videos_count_likes_view)
                        get_videos_count_likes_view=get_videos_count_likes_view.split(",")
                    except:
                        pass

                    try:
                        lik_video=[]
                        com_video=[]
                        view_video=[]
                        for ijk in get_videos_count_likes_view:
                            print(ijk)
                            print("==========")
                            if(("comments" in ijk)or("comment" in ijk)):
                                print("IJK IF",ijk)
                                com_video.append(ijk)
                            elif(("views" in ijk)or("view" in ijk)):
                                print("IJK ILIF",ijk)
                                view_video.append(ijk)
                            else:
                                lik_video.append(ijk)
                        print("lik_video---->com_video----->view_video",lik_video,com_video,view_video)
                    except:
                        pass

                    time.sleep(2)
                    try:
                        ele_click_most_relevent_button=driver.find_element(By.XPATH,"//div[@class='']//span[text()='Most relevant']")
                        ele_click_most_relevent_button.click()
                    except:
                        pass

                    time.sleep(2)
                    try:
                        ele_click_all_comments=driver.find_element(By.XPATH,"//span[text()='All comments']")
                        ele_click_all_comments.click()
                    except:
                        pass

                    time.sleep(3)
                    for i in range(0,10):
                        try:
                            ele_click_extra_comments_20=driver.find_elements(By.XPATH,"//*[contains(@id,'mount_0_0_')]//span[@class='alzwoclg lxowtz8q aeinzg81']")
                            print("Length of View More Comments Clicked",len(ele_click_extra_comments_20))
                            if(len(ele_click_extra_comments_20)>0):
                                ele_click_extra_comments=driver.find_element(By.XPATH,"//*[contains(@id,'mount_0_0_')]//span[@class='alzwoclg lxowtz8q aeinzg81']")
                                driver.execute_script("arguments[0].click();", ele_click_extra_comments)
                                print("Inside For Loop View More Comments Clicked")
                            else:
                                pass
                        except:
                            pass


                    time.sleep(3)
                    get_comments=driver.find_elements(By.XPATH,"//div[@class='om3e55n1 b0ur3jhr cgu29s5g mm05nxu8']//div[@class='om3e55n1']")
        #             print("len(get_comments)========>",len(get_comments))
                    if(len(get_comments)>0):
                        print("len(get_comments) Greater than Zero")
                        for jk in range(1,len(get_comments)+1):
                #             print("=================================")
                #             print(jk)
                            try:
                                get_person_url=driver.find_element(By.XPATH,f"(//div[@class='om3e55n1 b0ur3jhr cgu29s5g mm05nxu8']//div[@class='om3e55n1']//div[@class='lzubc330 qmqpeqxj e7u6y3za qwcclf47 nmlomj2f b6ax4al1 lxowtz8q fzsidkae om3e55n1']//span[@class='rse6dlih']//a){[jk]}").get_attribute("href")
        #                         print(get_person_url)
                            except:
                                pass

                            try:
                                person_name=driver.find_element(By.XPATH,f"(//div[@class='om3e55n1 b0ur3jhr cgu29s5g mm05nxu8']//div[@class='om3e55n1']//span[@class='fxk3tzhb']){[jk]}").text
        #                         print(person_name)
                            except:
                                pass

                            try:
                                get_commnets=driver.find_element(By.XPATH,f"(//div[@class='om3e55n1 b0ur3jhr cgu29s5g mm05nxu8']//div[@class='om3e55n1']//div[@class='e4ay1f3w r5g9zsuq aesu6q9g q46jt4gp']){[jk]}").text
        #                         print(get_commnets)
                            except:
                                pass

                            try:
            #                     print("============================================================")
                                datafr=datafr.append({'Name of the Page':get_page_name,'Url':current_url, 'Person Name':person_name, 'Person Url':get_person_url,'Likes Counts':lik_video,'Comments Counts':com_video,'Share Counts':view_video, 'Likes_Comments_Shares_Counts':get_videos_count_likes_view,'Comments':get_commnets},ignore_index=True)
                                print(datafr)
            #                     print("============================================================")
                            except Exception as e:
                                print(e)
                                pass
                    else:
                        print("len(get_comments) Lesser than Zero")
                        try:
                            get_person_url=''
        #                     print(get_person_url)
                        except:
                            pass

                        try:
                            person_name=''
        #                     print(person_name)
                        except:
                            pass

                        try:
                            get_commnets=''
        #                     print(get_commnets)
                        except:
                            pass

                        try:
        #                     print("============================================================")
                            datafr=datafr.append({'Name of the Page':get_page_name,'Url':current_url, 'Person Name':person_name, 'Person Url':get_person_url,'Likes Counts':lik_video,'Comments Counts':com_video,'Share Counts':view_video, 'Likes_Comments_Shares_Counts':get_videos_count_likes_view,'Comments':get_commnets},ignore_index=True)
                            print(datafr)
        #                     print("============================================================")
                        except Exception as e:
                            print(e)
                            pass

                else:           
                    print("*****************************INSIDE ELSE**************************************")
                    try:
                        get_page_name=driver.find_element(By.XPATH,"//div[@class='i0rxk2l3']//h2").text
            #             print("Page Name: ",get_page_name)
                    except:
                        pass
                    time.sleep(3)
                    try:
                        time.sleep(5)
                        get_commentslikes_share_count=[]
                        get_commentslikes_share_count1=driver.find_element(By.XPATH,"//div[@class='i85zmo3j rtxb060y alzwoclg k1z55t6l siu44isn oog5qr5w m8h3af8h rj0o91l8 kjdc1dyq p9ctufpz pvreidsc oxkhqvkx n68fow1o nch0832m mfycix9x']").text
                        get_commentslikes_share_count.append(get_commentslikes_share_count1)
                        print("get_commentslikes_share_count",get_commentslikes_share_count)
                    except:
                        pass
                    time.sleep(2)
                    try:
                        time.sleep(5)
                        su_likes=[]
                        su_likes1=driver.find_element(By.XPATH,"//div[@class='alzwoclg cqf1kptm cgu29s5g om3e55n1']//span[@class='o3hwc0lp lq84ybu9 hf30pyar oshhggmv lwqmdtw6']").text
                        su_likes.append(su_likes1)
                        print("Likes================>",su_likes)
                    except:
                        pass
                    time.sleep(2)
                    try:
                        su_cmt=[]
                        su_share=[]
                        su_cmt_share=driver.find_elements(By.XPATH,"//div[@class='alzwoclg cqf1kptm cgu29s5g om3e55n1']//div[@class='dkzmklf5']")
                        for su_i in su_cmt_share:
                            su_text_cmt_share_1=su_i.text
                            if(('comments' in su_text_cmt_share_1)or('comment' in su_text_cmt_share_1)):
                                su_cmt.append(su_text_cmt_share_1)
                            elif(('shares' in su_text_cmt_share_1)or('share' in su_text_cmt_share_1)):
                                su_share.append(su_text_cmt_share_1)
                        print("Comments Only====>",su_cmt)
                        print("su_share Only====>",su_share)
                    except:
                        pass

                    time.sleep(3)
                    for i in range(0,10):
                        try:
                            ele_click_extra_comments_20=driver.find_elements(By.XPATH,"//*[contains(@id,'mount_0_0_')]//span[@class='alzwoclg lxowtz8q aeinzg81']")
                            print("Length of View More Comments Clicked",len(ele_click_extra_comments_20))
                            if(len(ele_click_extra_comments_20)>0):
                                ele_click_extra_comments=driver.find_element(By.XPATH,"//*[contains(@id,'mount_0_0_')]//span[@class='alzwoclg lxowtz8q aeinzg81']")
                                driver.execute_script("arguments[0].click();", ele_click_extra_comments)
                                print("Inside For Loop View More Comments Clicked")
                            else:
                                pass
                        except:
                            pass

                    time.sleep(3)
                    get_comments=driver.find_elements(By.XPATH,"//div[@class='k0kqjr44']//div[@class='om3e55n1']")
        #             print(len(get_comments))
                    if(len(get_comments)>0):
        #                 print("Else====>len(get_comments) Greater than Zero")
                        for jk in range(1,len(get_comments)+1):
                            try:
                                get_person_url=driver.find_element(By.XPATH,f"(//div[@class='k0kqjr44']//div[@class='om3e55n1']//div[@class='lzubc330 qmqpeqxj e7u6y3za qwcclf47 nmlomj2f b6ax4al1 lxowtz8q fzsidkae om3e55n1']//a){[jk]}").get_attribute("href")
        #                         print(get_person_url)
                            except:
                                pass

                            try:
                                person_name=driver.find_element(By.XPATH,f"(//div[@class='k0kqjr44']//div[@class='om3e55n1']//span[@class='fxk3tzhb']){[jk]}").text
        #                         print(person_name)
                            except:
                                pass

                            try:
                                get_commnets=driver.find_element(By.XPATH,f"(//div[@class='k0kqjr44']//div[@class='om3e55n1']//div[@class='e4ay1f3w r5g9zsuq aesu6q9g q46jt4gp']){[jk]}").text
        #                         print(get_commnets)
                            except:
                                pass
                            try:
                #                     print("============================================================")
                                datafr=datafr.append({'Name of the Page':get_page_name,'Url':current_url, 'Person Name':person_name, 'Person Url':get_person_url,'Likes Counts':su_likes,'Comments Counts':su_cmt,'Share Counts':su_share, 'Likes_Comments_Shares_Counts':get_commentslikes_share_count,'Comments':get_commnets},ignore_index=True)
                                print(datafr)
                #                     print("============================================================")
                            except Exception as e:
                                print(e)
                                pass
                    else:
                        print("Else=====>len(get_comments) Lesser than Zero")
                        try:
                            get_person_url=''
        #                     print(get_person_url)
                        except:
                            pass

                        try:
                            person_name=''
        #                     print(person_name)
                        except:
                            pass

                        try:
                            get_commnets=''
        #                     print(get_commnets)
                        except:
                            pass
                        try:
            #                     print("============================================================")
                            datafr=datafr.append({'Name of the Page':get_page_name,'Url':current_url, 'Person Name':person_name, 'Person Url':get_person_url,'Likes Counts':su_likes,'Comments Counts':su_cmt,'Share Counts':su_share, 'Likes_Comments_Shares_Counts':get_commentslikes_share_count,'Comments':get_commnets},ignore_index=True)
                            print(datafr)
            #                     print("============================================================")
                        except Exception as e:
                            print(e)
                            pass

            else:
                print("Url already Exits",current_url)
                
        now = datetime.datetime.now()
        current_date=now.strftime("%d_%m_%Y")

        try:
            datafr.to_csv(f"FB_{get_page_name}_{current_date}.csv")
        except:
            pass
        
        try:
            datafr.to_csv(f"FB_{get_page_name}.csv")
        except:
            pass

        try:
             driver.stop_client()
        except:
            pass
        try:
            driver.close()
            driver.quit()
        except:
            pass

        return datafr
    
    data_gathered=facebook_data_extraction()
    print(data_gathered)
    return data_gathered


# In[39]:


df1=facebook_2()


# In[ ]:




