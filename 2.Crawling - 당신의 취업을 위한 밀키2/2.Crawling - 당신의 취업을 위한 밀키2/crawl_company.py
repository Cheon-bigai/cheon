from selenium import webdriver
from selenium.webdriver.common.by import By 
from selenium.common.exceptions import NoSuchElementException
from urllib.request import urlopen 
from bs4 import BeautifulSoup
import re
import pandas as pd
import os


#==========================================================================================================
# Web_crawling.py 폴더에서 저장된 채용공고.CSV 파일 
#
# 본인의 CSV 저장된 파일위치 경로 잘 확인해주세요. 
# 
#
# 채용공고.CSV 파일에서 데이터 불러오기


#===========================================================================================================

def crawl_company_info(data):
    print(data)
    """기업 정보를 크롤링하고 CSV 파일로 저장"""
    if not os.path.exists(data):
        return None
    company_df = pd.read_csv(data, encoding='utf-8-sig')
    # Selenium WebDriver 옵션 설정 (GPU 비활성화)
    options = webdriver.ChromeOptions()
    options.add_argument('--disable-gpu')
    driver = webdriver.Chrome(options=options)

    # 기업 정보 및 연봉 관련 데이터를 저장할 리스트 생성
    total_company_name_attr=[]

    # 기업 정보 XPath (HTML 요소의 경로) 지정
    xpath_dict = {
        '업력': '//*[@id="content"]/div/div[2]/section[2]/div[2]/div[2]/ul/li[1]/strong',          
        '기업형태': '//*[@id="content"]/div/div[2]/section[2]/div[2]/div[2]/ul/li[2]/div/button/strong',
        '사원수': '//*[@id="content"]/div/div[2]/section[2]/div[2]/div[2]/ul/li[3]/div/strong',
        '업종': '//*[@id="content"]/div/div[2]/section[2]/div[2]/div[2]/div/dl/div[1]/dd',
        '사업내용': '//*[@id="content"]/div/div[2]/section[2]/div[2]/div[2]/div/dl/div[4]/dd/p'
    }

    # 연봉 관련 XPath 지정
    xpath_sdict = {
        '평균연봉': '//*[@id="tab_avg_salary"]/div/div[1]/div[2]/p/em',                      
        '최저연봉': '//*[@id="tab_avg_salary"]/div/div[1]/div[2]/div/span[2]/em',
        '최고연봉': '//*[@id="tab_avg_salary"]/div/div[1]/div[2]/div/span[4]/em',
        '22년 대비': '//*[@id="tab_avg_salary"]/div/div[1]/div[3]/dl[1]/dd/em',
        '연봉신뢰도': '//*[@id="tab_avg_salary"]/div/div[1]/div[3]/dl[3]/dd/span'
    }

    # 매출액 및 영업이익 XPath 지정
    xpath_fdict = {
        '매출액': [
            '//*[@id="content"]/div/div[2]/section[1]/div/div[1]/div[1]/div/div[2]/div/div[2]/div[1]/div/span[2]',
            '//*[@id="content"]/div/div[2]/section[1]/div/div[1]/div[1]/div/div[2]/div/div[2]/div[2]/div/span[2]',
            '//*[@id="content"]/div/div[2]/section[1]/div/div[1]/div[1]/div/div[2]/div/div[2]/div[3]/div/span[2]',
            '//*[@id="content"]/div/div[2]/section[1]/div/div[1]/div[1]/div/div[2]/div/div[2]/div[4]/div/span[2]'
        ],
        '영업이익': [
            '//*[@id="content"]/div/div[2]/section[1]/div/div[1]/div[2]/div/div[2]/div/div[2]/div[1]/div/span[2]',
            '//*[@id="content"]/div/div[2]/section[1]/div/div[1]/div[2]/div/div[2]/div/div[2]/div[2]/div/span[2]',
            '//*[@id="content"]/div/div[2]/section[1]/div/div[1]/div[2]/div/div[2]/div/div[2]/div[3]/div/span[2]',
            '//*[@id="content"]/div/div[2]/section[1]/div/div[1]/div[2]/div/div[2]/div/div[2]/div[4]/div/span[2]'
        ]
    }

    # 각 기업의 정보를 크롤링하여 저장
    total_company_name_attr = []
    for idx_i in range(len(company_df['링크'])):
        company_name_attr = [company_df['회사명'][idx_i]]  # 기업명 저장
        driver.get(company_df['링크'][idx_i])  # 해당 기업의 페이지 접속

        try:
            # 기업 내부 페이지로 이동
            in_company_url = driver.find_element(By.XPATH, '//*[@id="content"]/div[3]/section[1]/div[1]/div[1]/div[1]/div[1]/a[1]').get_attribute('href')
            print(in_company_url)
            if in_company_url!=None:
                driver.get(in_company_url)
                html = urlopen(in_company_url)
                soup = BeautifulSoup(html, 'html.parser')
                # 기업 기본 정보 크롤링
                a=soup.select('ul.company_summary strong.company_summary_tit')
                list_company_attr5=['','','','','']
                for x in a:
                    name=x.text.strip()
                    if bool(re.search(r'.*\d+(년차)$',name)):
                        list_company_attr5[0]=name
                    elif bool(re.search(r'.*(업|기타)$',name)):
                        list_company_attr5[1]=name
                    elif bool(re.search(r'.*만원$',name))==False:
                        check=re.search(r'\d+',name)
                        a=re.search(r'\d+',name).group()
                        list_company_attr5[2]=a+'명'
                a=soup.select('div.company_details_group')
                for x in a:
                    if x.select_one('dt.tit').text=='업종':
                        list_company_attr5[3]=x.select_one('dt.tit').next_sibling.next_sibling.text
                    if x.select_one('dt.tit').text=='사업내용':
                        b=x.select_one('dt.tit').next_sibling.next_sibling.select_one('p')['title']
                        list_company_attr5[4]=b
                    else: pass
                company_name_attr.extend(list_company_attr5)            
                # 연봉 및 재무 정보 수집
                for category in ['연봉정보', '재무정보']:
                    for idx in range(1, 7):
                        path = f'//*[@id="content"]/div/div[1]/div/nav/ul/li[{idx}]/button'
                        try:
                            element = driver.find_element(By.XPATH, path)
                            if element.text == category:
                                detail_url = element.get_attribute('onclick').replace('window.location.href=', '').replace(';', '').strip("'")
                                company_url = f'https://www.saramin.co.kr{detail_url}' 
                                break
                            else:                                   
                                company_url=''
                        except NoSuchElementException:
                            continue
                    
                    if category == '연봉정보':
                        if company_url:
                            driver.get(company_url)
                            for key in xpath_sdict:
                                try:
                                    value = driver.find_element(By.XPATH, xpath_sdict[key]).text
                                except NoSuchElementException:
                                    value = ''
                                company_name_attr.append(value)
                        else:
                            company_name_attr.extend([''] * 5)
                    else:
                        if company_url:
                            driver.get(company_url)
                            for key in xpath_fdict:
                                for xpath in xpath_fdict[key]:
                                    try:
                                        value = driver.find_element(By.XPATH, xpath).text
                                    except NoSuchElementException:
                                        value = ''
                                    company_name_attr.append(value)
                        else:
                            company_name_attr.extend([''] * 8)
            else:pass
        except NoSuchElementException as e:
                print(e)
        total_company_name_attr.append(company_name_attr)

    driver.quit()
    # 데이터 프레임 생성 및 CSV 저장
    companyDF = pd.DataFrame(total_company_name_attr)
    companyDF.columns = ['기업명', '업력', '기업형태', '사원수', '업종', '사업내용', '평균연봉', '최저연봉', '최고연봉', '22년 대비 증감률', '연봉신뢰도', '매출액(2020)', '매출액(2021)', '매출액(2022)', '매출액(2023)', '영업이익(2020)', '영업이익(2021)', '영업이익(2022)', '영업이익(2023)']


    # 파일저장 한글/엑셀 모두 다 정상 
    # 
    # 기업정보 저장시 CSV 파일에서 원하는 데이터이름으로 저장해주세요.
    #
    companyDF.to_csv("기업정보.csv", encoding="utf-8-sig", index=False)













