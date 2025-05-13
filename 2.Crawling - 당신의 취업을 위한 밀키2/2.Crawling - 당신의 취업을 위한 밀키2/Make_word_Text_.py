from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
import re
import pandas as pd
import time
import os

def extract_preference_keywords(file_path, use_gui=False):
    """채용공고 CSV에서 우대사항과 마감기한을 크롤링하고 저장"""

    print(f"🔍 [mkw.py] 전달받은 파일 경로: {file_path}")  # ✅ 파일 경로 확인

    if not file_path or not os.path.exists(file_path):
        if use_gui:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(None, "경고", "먼저 파일을 업로드하세요!")  # GUI 모드일 때 경고창 출력
        print("❌ 오류: 파일이 존재하지 않거나 선택되지 않았습니다.")
        return
    
    print(f"✅ 파일 로드 성공: {file_path}")  # ✅ 파일 로드 확인

    driver = webdriver.Chrome()
    extracted_text_list = []  # ✅ 리스트로 저장하여 줄바꿈 적용

    ai_info_df = pd.read_csv(file_path)
    
    if '링크' not in ai_info_df.columns:
        if use_gui:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(None, "오류", "CSV 파일에 '링크' 열이 없습니다.")
        print("❌ 오류: CSV 파일에 '링크' 열이 없습니다.")
        driver.quit()
        return

    job_link_list = ai_info_df['링크'].dropna().tolist()  # 기업 공고 URL 리스트
    
    for job_url in job_link_list:
        driver.get(job_url)  # 기업 공고 페이지 접속
        time.sleep(2)

        try:
            iframe = driver.find_element(By.ID, "iframe_content_0")
            driver.switch_to.frame(iframe)

            iframe_html = driver.page_source
            soup = BeautifulSoup(iframe_html, 'html.parser')

            for script in soup(["script", "style"]):
                script.extract()

            # ✅ 우대사항 추출
            preferences = soup.find_all(string=re.compile('우대'))
            for preference in preferences:
                parent = preference.find_parent()
                if parent:
                    grandparent = parent.find_parent()
                    if grandparent:
                        text = grandparent.get_text(strip=True)
                        if text:
                            extracted_text_list.append(f"🔹 우대사항: {text}")  # ✅ 보기 좋게 가공
                        next_sibling = grandparent.find_next_sibling()
                        if next_sibling:
                            sibling_text = next_sibling.get_text(strip=True)
                            if sibling_text:
                                extracted_text_list.append(f"🔹 추가 정보: {sibling_text}")  # ✅ 추가 정보도 포함
                    else:
                        extracted_text_list.append(f"🔹 우대사항: {parent.get_text(strip=True)}")

            # ✅ 채용 마감기한 추출
            closing_date = soup.find(string=re.compile(r'마감기한|채용 마감|접수 마감|마감 일자'))
            closing_date_text = "미정"

            if closing_date:
                closing_date_text = closing_date.find_parent().get_text(strip=True)

            extracted_text_list.append(f"📌 채용 마감기한: {closing_date_text}")

        except Exception as e:
            print(f"❌ 우대사항 또는 마감기한을 찾을 수 없음: {job_url}, 오류: {e}")

    driver.quit()

    # ✅ 중복 제거 및 정리 후 파일 저장
    extracted_text_list = list(set(extracted_text_list))  # 중복 제거
    extracted_text = "\n".join(extracted_text_list)  # ✅ 보기 좋게 줄바꿈 처리
    output_file = "extracted_preference_keywords.txt"

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(extracted_text)

    if use_gui:
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(None, "완료", "우대사항 키워드 및 마감기한 추출 완료!\n저장된 파일: extracted_preference_keywords.txt")

    print(f"✅ 우대사항 키워드 추출 완료: {output_file}")
    return output_file  # ✅ 파일 경로 반환
