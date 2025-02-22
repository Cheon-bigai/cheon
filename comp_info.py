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
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from urllib.request import urlopen, Request
import re
import pandas as pd


driver = webdriver.Chrome()
keyword = '클라우드'
start_page = 1
end_page = 1


keyword_lst = []
salary_info_dict = {}
company_attr_total=[]


xpath_sdict = {'평균연봉': '//*[@id="tab_avg_salary"]/div/div[1]/div[2]/p/em',                      
                '최저연봉': '//*[@id="tab_avg_salary"]/div/div[1]/div[2]/div/span[2]/em',
                '최고연봉': '//*[@id="tab_avg_salary"]/div/div[1]/div[2]/div/span[4]/em',
                '22년 대비': '//*[@id="tab_avg_salary"]/div/div[1]/div[3]/dl[1]/dd/em',
                '연봉신뢰도': '//*[@id="tab_avg_salary"]/div/div[1]/div[3]/dl[3]/dd/span'}


#주소 열기
for i in range(start_page,end_page+1):
    driver.get(f'https://www.saramin.co.kr/zf_user/search/recruit?searchType=search&searchword={keyword}\
    &company_cd=0%2C1%2C2%2C3%2C4%2C5%2C6%2C7%2C9%2C10&panel_type=&search_optional_item=y&search_done=y&panel_count=y&\
    preview=y&recruitPage={i}')
    time.sleep(3)
    # 해당 주소의 기업의 링크 저장 => 리스트에서 개별적으로 href 속성 가져오기
    job_links = driver.find_elements("css selector", "h2.job_tit a")
    job_link_list = [job.get_attribute("href") for job in job_links]
    #기업의 주소들 가져오기
    # 각 공고 링크에 들어가서 정보 크롤링
    for job_url in job_link_list:                                     # 기업 공고 페이지 들어가기
        driver.get(job_url)  # 공고 페이지 방문
        # time.sleep(2)  # 페이지 로딩 대기 (필요에 따라 조절)


        job_links = driver.find_element("css selector", "div.title_inner a").get_attribute('href')
        if job_links!=None:
            driver.get(job_links)
        # job_link_list = [job.get_attribute("href") for job in job_links] #기업의 주소들 가져오기
        # for i in job_link_list:
            # print(i)
            # 기업명
            try:
                company_name = driver.find_element(By.CSS_SELECTOR, "div.box_title h1.tit_company").text.strip()
            except NoSuchElementException:
                company_name = "알 수 없음"
            salary_data = []  
            company_attr=[company_name]
            for x in ['연봉정보']:
                for y in range(1,7):
                    path=f'//*[@id="content"]/div/div[1]/div/nav/ul/li[{y}]/button'
                    try:
                        a = driver.find_element(By.XPATH,path)
                        if a.text==x:
                            detail_url = a.get_attribute('onclick').replace('window.location.href=','').replace(';','').strip("'")
                            company_url=r'https://www.saramin.co.kr'+detail_url
                            break
                        else: company_url=''
                    except NoSuchElementException: company_url=''
                   
                if x=='연봉정보':
                    if len(company_url)>=1:
                        print(company_url)
                        driver.get(company_url)
                    # 5개정보
                        for x in range(len(xpath_sdict)):
                            try:
                                attr=driver.find_element(By.XPATH,xpath_sdict[list(xpath_sdict.keys())[x]]).text
                            except NoSuchElementException:
                                attr=''
                            company_attr.append(attr)
                    else: company_attr.extend(['']*5)
                else: pass
            company_attr_total.append(company_attr)
print(company_attr_total)      
driver.quit()
 




























            # driver.get(i)
            # html = driver.page_source
            # soup=BeautifulSoup(html,'html.parser')
            # a=soup.select('ul.company_summary strong.company_summary_tit')
            # list_company_attr5=['','','','','']
            # for x in a:
            #     name=x.text.strip()
            #     if bool(re.search(r'.*\d+(년차)$',name)):
            #         list_company_attr5[0]=name
            #     elif bool(re.search(r'.*(업|기타)$',name)):
            #         list_company_attr5[1]=name
            #     elif bool(re.search(r'.*만원$',name))==False:
            #         check=re.search(r'\d+',name)
            #         a=re.search(r'\d+',name).group()
            #         list_company_attr5[2]=a+'명'
            # a=soup.select('div.company_details_group')
        
        
        
        
      
      
      
      
      
      
      
      
      
      
      
      
      
      
        
        
#         job_links = driver.find_elements("css selector", "div.title_inner a")
#         job_link_list = [job.get_attribute("href") for job in job_links] #기업의 주소들 가져오기

#         try:
#             company_link_element = driver.find_element(By.CSS_SELECTOR, "div.title_inner a")
#             company_link = company_link_element.get_attribute("href")

#         # URL이 올바른지 검증
#             if not company_link or not company_link.startswith("http"):
#                 print(f"잘못된 company_link 값: {company_link}")
#             else:
#                 driver.get(company_link)  # 유효한 경우에만 이동

#             company_info1 = driver.find_element(By.CSS_SELECTOR, 'ul.company_summary').text
#             print(company_info1)
#             print('-'*50)
            
#         except Exception as e:
#             print(f"오류 발생: {e}")
            

# driver.quit()