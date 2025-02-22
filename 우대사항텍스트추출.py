from bs4 import BeautifulSoup
import requests
from konlpy.tag import Okt
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import time
import platform
import numpy as np
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd


driver = webdriver.Chrome()
keyword = '클라우드' # 원하는 키워드 넣으세요
start_page = 1      # 원하는 시작 페이지
end_page = 1        # 원하는 끝 펭지ㅣ

keyword_lst = []

#주소 열기
for i in range(start_page,end_page+1):
    driver.get(f'https://www.saramin.co.kr/zf_user/search/recruit?searchType=search&searchword={keyword}\
    &company_cd=0%2C1%2C2%2C3%2C4%2C5%2C6%2C7%2C9%2C10&panel_type=&search_optional_item=y&search_done=y&panel_count=y&\
    preview=y&recruitPage={i}')


    # 해당 주소의 기업의 링크 저장 => 리스트에서 개별적으로 href 속성 가져오기
    job_links = driver.find_elements("css selector", "h2.job_tit a")
    job_link_list = [job.get_attribute("href") for job in job_links] #기업의 주소들 가져오기


    # 각 공고 링크에 들어가서 정보 크롤링
    for job_url in job_link_list:                                     # 기업 공고 페이지 들어가기
        driver.get(job_url)  # 공고 페이지 방문
        time.sleep(2)  # 페이지 로딩 대기 (필요에 따라 조절)


        try:      
            company_name = driver.find_element("css selector", "div.jv_header div.title_inner a") # 채용기업명
            job_summary = driver.find_element("css selector", "div.jv_cont.jv_summary") # 개괄 채용정보 (경력,학력,근무형태)
           
            text_content1 = company_name.text  
            text_content2 = job_summary.text  
             
             
            #print('='*50)
            #print(f'[기업명] :{text_content1}') # (1)defautl
            #print(f'[기업에 대한 정보] :{text_content2}') # (2) 기업 핵심정보
#--------------------------------------------------------------------------------------
            #채용정보 > 채용 글 내용


            #iframe으로 전환
            iframe = driver.find_element(By.ID, "iframe_content_0")
            driver.switch_to.frame(iframe)


            #내부 전체 텍스트 가져오기
            iframe_text = driver.find_element(By.TAG_NAME, "body").text.strip()
            #print(iframe_text)                     #(3) 기업 채용 글 내용
            
#-----------------------------------------------------------------------------------
#(1) 텍스트 파일로 만들고 싶을 시 작동            
            #keyword_lst.append(iframe_text)
#------------------------------------------------------------------------------------
#(2 csv 파일로 만들고 싶을 시 작동
            keyword_lst.append([company_name, job_summary, iframe_text])

    
        except:
            print(f"해당 페이지에서 wrap_jv_cont를 찾을 수 없음: {job_url}")
    print(keyword_lst)


driver.quit()


##------------------------------------------------------------------------------------
#(1)텍스트 파일로 만들고 싶을시 같이 작동
# with open("채용정보.txt", "w", encoding="utf-8") as f:
#     for job_text in keyword_lst:
#         f.write(job_text)  # 각 항목을 줄바꿈하여 저장

# print("채용 정보가 '채용정보.txt' 파일로 저장되었습니다.")
##------------------------------------------------------------------------------------
#(2) csv 파일로 만들고 싶을 시 같이 작동
df = pd.DataFrame(keyword_lst, columns=['기업명', '채용 정보', '채용 상세'])
df.to_csv("기업_채용정보.csv", index=False, encoding='utf-8-sig')