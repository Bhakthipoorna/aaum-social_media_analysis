#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
# path="Downloads\chromedriver.exe"
option.add_argument("--disable-infobars")
option.add_argument("start-maximized")
option.add_argument("--incognito")
option.add_argument("--headless")
# option.add_argument("--window-size=1382,754")
option.add_argument("--window-size=1920,1080")
# Pass the argument 1 to allow and 2 to block
option.add_experimental_option("prefs", { 
    "profile.default_content_setting_values.notifications": 2
})


# In[ ]:


def instagram():
    
    # 
    def login_credential():
        insta_url="https://www.instagram.com/accounts/login/"
        # username="7010413012"
        # password="aaum123"

        # username="sundar_sp_0002"
        # password="aaum@1234"
        username="poyyinselvan1947@gmail.com"
        password="Ponniyin@567"

        page_name1="https://www.instagram.com/madrastalkies/"
        
        file_name="Instagram_data_till_Sep_03.csv"
        return insta_url,username,password,page_name1,file_name

    instagram_url,username,password,page_name,file_name=login_credential()[0],login_credential()[1],login_credential()[2],login_credential()[3],login_credential()[4]
    print(instagram_url)
    print(username)
    print(password)
    print(page_name)
    print(file_name)
    
    def previous_data_load(file_name):
        all_data=pd.read_csv(file_name,index_col=0)
        all_data.drop('_id',axis=1,inplace=True)
        list_of_all_old_url=list(all_data['Image_Url'].unique())
        print("Length of Old Url Unique Counts :",len(list_of_all_old_url))
        return list_of_all_old_url

    previous_data=previous_data_load(file_name)
    print(previous_data)
    
    def instagram_extracting_data():
        df1 = pd.DataFrame()

        driver = webdriver.Chrome(options=option, executable_path=path)
    #     driver.maximize_window()
        time.sleep(2)

        driver.delete_all_cookies()

        time.sleep(3)
        driver.get(instagram_url)

        time.sleep(2)
        element_username=driver.find_element(By.NAME,"username")
        element_password=driver.find_element(By.NAME,"password")
        element_login_button=driver.find_element(By.XPATH,"//div[text()='Log In']")

        time.sleep(2)
        element_username.send_keys(username)
        element_password.send_keys(password)
        element_login_button.click()
        time.sleep(5)
        try:
            driver.find_element(By.CLASS_NAME,"cmbtv").click()
        except:
            pass
        print("Inside Insta")
        time.sleep(5)
        try:
            driver.get(page_name)
        except:
            pass
        time.sleep(3)
        for ii in range(10):
            time.sleep(2)
#             print(ii)
        time.sleep(5)


        try:
            name_of_the_page=driver.find_element(By.XPATH,"//div[@class='_aa_m']//h2").text
            print("Name of the Page :",name_of_the_page)
        except:
            pass
        
        try:
            page_followers_count=driver.find_element(By.XPATH,"//*[contains(@id,'mount_0_0_')]/div/div/div/div[1]/div/div/div/div[1]/section/main/div/header/section/ul/li[2]/a/div").text
            print("page_followers_count :",page_followers_count)
        except:
            pass

        time.sleep(3)
        try:
            get_image_url=driver.find_elements(By.XPATH,"//article[@class='_aayp']//a")
            len_images=len(get_image_url)
#             print("len_images---------->",len_images)
        except:
            pass

        a = ActionChains(driver)

        d={}

        time.sleep(2)
        for lo in range(1,len_images+1):
            time.sleep(3)
            print(lo)
            try:
                m=driver.find_element(By.XPATH,f"(//article[@class='_aayp']//a){[lo]}")
                a.move_to_element(m).perform()
            except:
                pass
            time.sleep(3)
            try:
                time.sleep(3)
                img=driver.find_element(By.XPATH,f"(//article[@class='_aayp']//a){[lo]}").get_attribute('href')
    #             print(img)
                time.sleep(2)
                total_likes=driver.find_element(By.XPATH,"(//li[@class='_abpm'])[1]").text
    #             print(total_likes)
                time.sleep(2)
                total_comments=driver.find_element(By.XPATH,"(//li[@class='_abpm'])[2]").text
#                 print(total_comments)
    #             print("=====================================")
            except:
                pass

            try:
                df=pd.DataFrame({'Name of the Page':name_of_the_page,'Image_Url':img,'Total Likes':total_likes,'Total Comments':total_comments},index=[0])
    #             print(df)
            except:
                pass

    #         df1 = pd.concat([df1,df])

            try:
                for iter_img,iter_comments_counts in zip(df['Image_Url'],df['Total Comments']):
                    print(iter_img,iter_comments_counts)
                    d[iter_img]=iter_comments_counts
            except:
                pass

    #     print("d==============***********==================>",d)
        clean_dict = {str(key).strip(): re.sub(r"[^a-zA-Z0-9 ]", "", str(item)) for key, item in d.items()}
        print("clean_dict:",clean_dict)

        for url_image1,counting in clean_dict.items():
    #         print(url_image1,int(counting)//4)
            try:
                driver.get(url_image1)
            except:
                pass

            if(url_image1 not in previous_data):
                print("Current Available New URL",url_image1)
                time.sleep(2)
                try:
                    name_of_the_page=driver.find_element(By.XPATH,"//div[@class='_aaqt']//a").text
                except:
                    pass

                try:
                    date_of_post=driver.find_element(By.XPATH,"//time[@class='_aaqe']").get_attribute("datetime")
                    time_of_post=driver.find_element(By.XPATH,"//time[@class='_aaqe']").text
        #             print(date_of_post,"---------->",time_of_post)
                except:
                    pass
                try:
                    get_views_or_likes=driver.find_element(By.XPATH,"//section[@class='_ae5m _ae5n _ae5o']").text
                except:
                    pass

                time.sleep(3)
                for loo in range(1,50):
                    if(loo>0):
                        try:
                            time.sleep(2)
#                             print("loo--->",loo)
                            clicking_plus_button=driver.find_elements(By.XPATH,"//*/div/div[1]/div/div[1]/div/div/div[1]/div[1]/section/main/div[1]/div[1]/article/div/div[2]/div/div[2]/div[1]/ul/li/div/button")
                            l=len(clicking_plus_button)
#                             print("L--------->",l)
                        except:
                            pass

                        if(l>0):
                            try:
                                time.sleep(2)
                                click_plus_button=driver.find_element(By.XPATH,"//*/div/div[1]/div/div[1]/div/div/div[1]/div[1]/section/main/div[1]/div[1]/article/div/div[2]/div/div[2]/div[1]/ul/li/div/button")
                                click_plus_button.click()
#                                 print("Inside IF--------->",l)
                            except:
                                pass

                time.sleep(2)
                click_ele_reply_comments=driver.find_elements(By.XPATH,"//button[@class='_acan _acao _acas']")
#                 print(len(click_ele_reply_comments))
                try:
                    for k in click_ele_reply_comments:
                        k.click()
                except:
                    pass

                time.sleep(2)
                try:
                    get_all_text=driver.find_elements(By.XPATH,"//div[@class='_a9zm']//div[@class='_a9zr']")
#                     print("get_all_text","-------->",len(get_all_text))
                except:
                    pass

                time.sleep(2)
                for jk in range(1,len(get_all_text)+1):
                #     print(i)
                    time.sleep(1)
                    try:
                        get_peron_name=driver.find_element(By.XPATH,f"(//div[@class='_ae5q _ae5r _ae5s']//ul)[{jk}]//h3").text
                #         print(get_peron_name)
                    except:
                        pass

                    try:
                        get_person_url=driver.find_element(By.XPATH,f"(//div[@class='_ae5q _ae5r _ae5s']//ul)[{jk}]//a").get_attribute('href')
                #         print(get_person_url)
                    except:
                        pass

                    try:
                        get_person_message=driver.find_element(By.XPATH,f"(//div[@class='_ae5q _ae5r _ae5s']//ul)[{jk}]//span[@class='_aacl _aaco _aacu _aacx _aad7 _aade']").text
                #         print(get_person_message)
                    except:
                        pass
                    try:
                        print(get_peron_name,"----->",get_person_url,"----------->",get_person_message)
                    except:
                        pass

                    try:
                        df=pd.DataFrame({'Name of the Page':name_of_the_page,'Image_Url':url_image1,'Likes Counts':get_views_or_likes,'Comments Counts':counting,'Date of Post':date_of_post,'Time of Post':time_of_post,"Likes or Views":get_views_or_likes,"Commented_Person_name":get_peron_name,"Person Url":get_person_url,"Comments":get_person_message,'Follower Count':page_followers_count},index=[0])
                        df1 = pd.concat([df1,df])  
                        print("=====================")
                        print(df)
                        print("====================")
                    except Exception as e:
                        print(e)
                        pass
            else:
                print("Current New URL are Already Exits ",url_image1)
                
        now = datetime.datetime.now()
        current_date=now.strftime("%d_%m_%Y")

        try:
            df1.to_csv(f"Instagram_{name_of_the_page}_{current_date}.csv")
        except:
            pass
        
        try:
            df1.to_csv(f"Instagram_{name_of_the_page}.csv")
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

        return df1
   
    data_gathered=instagram_extracting_data()
    
    print(data_gathered)
    return data_gathered


# In[ ]:


df1=instagram()


# In[ ]:


df1

