from bs4 import BeautifulSoup  # HTML 파싱을 위한 라이브러리
from urllib.request import Request, urlopen  # 웹 요청을 위한 라이브러리
from urllib.parse import quote  # URL 인코딩을 위한 라이브러리
import datetime  # 날짜를 다루기 위한 라이브러리
import pandas as pd  # 데이터프레임을 다루기 위한 라이브러리
import csv  # CSV 파일 저장을 위한 라이브러리

#====================================================================================================
# 검색어 설정
# 사람인에서 [채용공고 검색어 ]  입력하고 싶은 검색어 입력

keyword = '# 원하는 검색어 입력'  # 원하는 검색어 입력
search_word = quote(keyword)  # 검색어를 URL 인코딩 (한글 깨짐 방지)

#====================================================================================================


# 총 페이지 수를 가져오는 함수
def get_total_pages(url):
    urlrequest = Request(url, headers={'User-Agent': 'Mozilla/5.0'})  # 웹 요청 생성 (User-Agent 설정으로 차단 방지)
    html = urlopen(urlrequest)  # URL 열기
    soup = BeautifulSoup(html, 'html.parser')  # HTML 파싱

    # 페이지네이션 영역 찾기
    pagination = soup.find('div', {'class': 'pagination'})
    total_pages = 1  # 기본 페이지 수는 1 (페이지네이션이 없는 경우 대비)
    if pagination:
        page_numbers = pagination.find_all('a', {'class': 'page'})  # 페이지 번호가 들어 있는 링크 검색
        if page_numbers:
            total_pages = int(page_numbers[-1].text)  # 마지막 페이지 번호가 전체 페이지 수

    return total_pages

# 채용 정보를 수집하는 함수
def get_sector_data(url, keyword, sector_list, max_pages):
    try:
        for page in range(1, max_pages + 1):  # 1페이지부터 max_pages까지 순회
            paginated_url = url + f'&recruitPage={page}&recruitSort=relation&recruitPageCount=1000'  # 페이지네이션 적용
            urlrequest = Request(paginated_url, headers={'User-Agent': 'Mozilla/5.0'})  # 웹 요청 생성
            html = urlopen(urlrequest)  # URL 열기
            soup = BeautifulSoup(html, 'html.parser')  # HTML 파싱

            # 채용 공고 목록 가져오기
            job_items = soup.find_all('div', {'class': 'item_recruit'})

            # 각 채용 공고에서 정보 추출
            for job in job_items:
                area_job = job.find('div', {'class': 'area_job'})  # 직무 정보가 있는 영역
                company_tag = job.find('div', {'class': 'area_corp'})  # 회사 정보가 있는 영역
                
                # 채용 공고 제목 및 링크 가져오기
                job_title_tag = area_job.find('h2', {'class': 'job_tit'}).find('a')
                
                # 직무 조건 가져오기
                job_condition = area_job.find('div', {'class': 'job_condition'})
                job_info = job_condition.find_all('span')  # 위치, 경력, 학력, 급여 등의 정보 포함
                
                # 검색어와 관련된 채용 공고인지 확인
                if job_title_tag and 'title' in job_title_tag.attrs:
                    job_title = job_title_tag['title'].strip()  # 채용 공고 제목
                    job_link = 'https://www.saramin.co.kr' + str(job_title_tag.attrs['href'])  # 채용 공고 링크
                    
                    company_name = company_tag.find('strong', {'class': 'corp_name'}).text.strip()  # 회사명
                    
                    # 직무 조건 정보 추출 (없을 경우 기본값 N/A)
                    location = job_info[0].text.strip() if len(job_info) > 0 else 'N/A'
                    career = job_info[1].text.strip() if len(job_info) > 1 else 'N/A'
                    education = job_info[2].text.strip() if len(job_info) > 2 else 'N/A'
                    salary = job_info[3].text.strip() if len(job_info) > 3 else 'N/A'
                    
                    # 채용 공고 정보를 딕셔너리 형태로 저장
                    job_details = {
                        '회사명': company_name,
                        '채용공고사항': job_title,
                        '링크': job_link,
                        '지역': location,
                        '경력': career,
                        '학력': education,
                        '월급': salary
                    }
                    
                    # 리스트에 추가
                    sector_list.append(job_details)
    except Exception as e:
        print(e)

# CSV 파일 만들기
def make_CSV (search_word):
    # 검색 URL 생성
    url = f'https://www.saramin.co.kr/zf_user/search?' \
                f'search_area=main&search_done=y&search_optional_item=n&searchType=search&' \
                f'searchword={search_word}'

    # 채용 정보 저장을 위한 리스트 생성
    sector_list = []

    # 오늘 날짜 가져오기
    today = datetime.datetime.now().strftime('%y-%m-%d')

    # 총 페이지 수 가져오기
    total_pages = get_total_pages(url)
    print(f'총 페이지 수: {total_pages}, 날짜: {today}, 검색어: {search_word}')

    # 채용 정보 수집
    get_sector_data(url, search_word, sector_list, total_pages)

    # 데이터프레임 생성
    job_saramin_df = pd.DataFrame(sector_list)

    # 파일명 설정 (검색어 기반)
    file =f'채용공고.csv' 

    # CSV 파일로 저장 (UTF-8 인코딩)
    job_saramin_df.to_csv(file, encoding="utf-8-sig", index=False)
    return file

    # 결과 출력
    # print(job_saramin_df)

# 함수실행 전에 
# make_CSV(search_word)
