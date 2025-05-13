from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QLineEdit,QMessageBox,QTextEdit
import subprocess
import sys
import urllib.parse
import os
from crawl_company import crawl_company_info
from vizual_function import *
from crawl_recruitment import *
from Make_word_Text_ import *

class RecruitmentApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle("채용 공고 GUI")
        self.selected_file = None
        self.data = None  # 선택된 파일의 데이터 저장 변수
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.initUI()

    def initUI(self):
        self.clear_layout()
        
        self.search_btn = QPushButton("검색어 입력")
        self.search_btn.clicked.connect(self.show_search_page)
        self.layout.addWidget(self.search_btn)

        self.company_info_btn = QPushButton("회사 정보")
        self.company_info_btn.clicked.connect(self.show_company_info_page)
        self.layout.addWidget(self.company_info_btn)

    def show_search_page(self):
        self.clear_layout()

        self.layout.addWidget(QLabel("검색어 입력:"))
        self.search_entry = QLineEdit()
        self.layout.addWidget(self.search_entry)

        search_execute_btn = QPushButton("검색 실행")
        search_execute_btn.clicked.connect(lambda: self.run_search(self.search_entry.text()))
        self.layout.addWidget(search_execute_btn)
        
        next_btn = QPushButton("다음")
        next_btn.clicked.connect(self.show_file_page)
        self.layout.addWidget(next_btn)

        back_btn = QPushButton("뒤로")
        back_btn.clicked.connect(self.initUI)
        self.layout.addWidget(back_btn)
    
    def show_file_page(self):
        self.clear_layout()

        file_btn = QPushButton("파일 찾기")
        file_btn.clicked.connect(self.open_file1)
        self.layout.addWidget(file_btn)

        company_info_btn = QPushButton("기업 정보 분석")
        company_info_btn.clicked.connect(self.generate_company_info)
        self.layout.addWidget(company_info_btn)

        back_btn = QPushButton("뒤로")
        back_btn.clicked.connect(self.show_search_page)
        self.layout.addWidget(back_btn)

    def show_company_info_page(self):
        self.clear_layout()

        summary_btn = QPushButton("종합 정보")
        summary_btn.clicked.connect(self.show_summary_page)
        self.layout.addWidget(summary_btn)

        preference_btn = QPushButton("우대 사항")
        preference_btn.clicked.connect(self.show_preference_page)
        self.layout.addWidget(preference_btn)

        back_btn = QPushButton("뒤로")
        back_btn.clicked.connect(self.initUI)
        self.layout.addWidget(back_btn)

    def show_summary_page(self):
        self.clear_layout()

        file1_btn = QPushButton("파일 1 업로드")
        file1_btn.clicked.connect(self.open_file1)
        self.layout.addWidget(file1_btn)

        file2_btn = QPushButton("파일 2 업로드")
        file2_btn.clicked.connect(self.open_file2)
        self.layout.addWidget(file2_btn)

        visualize_btn = QPushButton("시각화 실행")
        visualize_btn.clicked.connect(self.visualize_data)
        self.layout.addWidget(visualize_btn)

        back_btn = QPushButton("뒤로")
        back_btn.clicked.connect(self.show_company_info_page)
        self.layout.addWidget(back_btn)

    def show_preference_page(self):
        """우대사항 키워드 추출 페이지"""
        self.clear_layout()

        file_btn = QPushButton("채용공고 파일 업로드")
        file_btn.clicked.connect(self.open_file1)
        self.layout.addWidget(file_btn)

        extract_btn = QPushButton("우대사항 키워드 추출")
        extract_btn.clicked.connect(self.run_extract_preference_keywords)
        self.layout.addWidget(extract_btn)

        # ✅ 직접 키워드 파일 업로드 버튼 추가
        upload_txt_btn = QPushButton("추출된 키워드 파일 업로드")
        upload_txt_btn.clicked.connect(self.upload_keyword_file)
        self.layout.addWidget(upload_txt_btn)

        # ✅ 우대사항 키워드를 표시할 QTextEdit 추가
        self.keyword_text_edit = QTextEdit()
        self.keyword_text_edit.setReadOnly(True)  # 읽기 전용
        self.layout.addWidget(self.keyword_text_edit)

        back_btn = QPushButton("뒤로")
        back_btn.clicked.connect(self.show_company_info_page)
        self.layout.addWidget(back_btn)

    def upload_keyword_file(self):
        """사용자가 추출된 키워드 파일을 직접 업로드하여 보기"""
        file_path, _ = QFileDialog.getOpenFileName(self, "우대사항 키워드 파일 선택", "", "Text Files (*.txt);;All Files (*)")

        if file_path:
            print(f"📂 선택된 키워드 파일: {file_path}")  
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    keywords_text = file.readlines()  # ✅ 한 줄씩 리스트로 가져오기

                    # ✅ 각 줄에 "🔹 우대사항: " 추가하여 보기 좋게 출력
                    formatted_text = "\n".join([f"🔹 우대사항: {line.strip()}" for line in keywords_text if line.strip()])

                    if not formatted_text:
                        self.keyword_text_edit.setPlainText("❌ 파일이 비어 있습니다.")
                        print("❌ 파일이 비어 있음")
                    else:
                        self.keyword_text_edit.setPlainText(f"📄 우대사항 키워드:\n\n{formatted_text}")
                        print(f"✅ 파일 내용 표시 완료:\n{formatted_text}")

            except Exception as e:
                QMessageBox.critical(self, "오류", f"파일을 읽는 중 오류 발생: {e}")
                print(f"❌ 파일 읽기 오류: {e}")


    def run_extract_preference_keywords(self):
        """우대사항, 우대기간, 마감기한 추출 및 GUI에 표시"""
        if not self.selected_file:
            QMessageBox.warning(self, "경고", "먼저 파일을 업로드하세요!")
            print("❌ 오류: `self.selected_file`이 설정되지 않음")
            return

        file_path = str(self.selected_file).strip()
        print(f"📂 [GUI] 최종 전달할 파일 경로: {file_path}")

        if not os.path.exists(file_path):
            QMessageBox.critical(self, "오류", f"파일이 존재하지 않습니다:\n{file_path}")
            print("❌ 오류: 파일이 존재하지 않음")
            return

        print("🚀 `extract_preference_keywords()` 실행!")

        try:
            output_file = extract_preference_keywords(file_path)  

            if output_file:
                QMessageBox.information(self, '완료', f'우대사항 키워드가 {output_file}로 저장되었습니다!')

                # ✅ 생성된 파일에서 내용 읽어와서 GUI에 표시 (QTextEdit 사용)
                with open("extracted_preference_keywords.txt", "r", encoding="utf-8") as file:
                    keywords_text = file.read()
                    self.keyword_text_edit.setPlainText(f"📄 추출된 정보:\n\n{keywords_text}")  # ✅ QTextEdit에 적용

        except Exception as e:
            QMessageBox.critical(self, "오류", f"키워드 추출 중 오류 발생: {e}")
            print(f"❌ 오류 발생: {e}")

    def clear_layout(self):
        while self.layout.count():
            item = self.layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def open_file1(self):
        self.selected_file, _ = QFileDialog.getOpenFileName(self, "파일 선택", "", "CSV Files (*.csv);;All Files (*)")

        if self.selected_file:
            decoded_path = urllib.parse.unquote(self.selected_file)  # URL 인코딩 해제
            print(f"📂 선택된 파일 (원본): {self.selected_file}")
            print(f"📂 선택된 파일 (디코딩 후): {decoded_path}")

            # 원본과 디코딩된 파일 중 존재하는 파일을 선택
            if os.path.exists(decoded_path):
                self.selected_file1 = decoded_path
            elif os.path.exists(self.selected_file):
                self.selected_file1 = self.selected_file
            else:
                print("❌ 파일이 존재하지 않습니다!")
                return

            print(f"✅ 최종 파일 경로: {self.selected_file}")
            self.data = self.selected_file  # 데이터 저장
            
    def open_file2(self):
        self.selected_file, _ = QFileDialog.getOpenFileName(self, "파일 선택", "", "CSV Files (*.csv);;All Files (*)")

        if self.selected_file:
            decoded_path = urllib.parse.unquote(self.selected_file)  # URL 인코딩 해제
            print(f"📂 선택된 파일 (원본): {self.selected_file}")
            print(f"📂 선택된 파일 (디코딩 후): {decoded_path}")

            # 원본과 디코딩된 파일 중 존재하는 파일을 선택
            if os.path.exists(decoded_path):
                self.selected_file2 = decoded_path
            elif os.path.exists(self.selected_file):
                self.selected_file2 = self.selected_file
            else:
                print("❌ 파일이 존재하지 않습니다!")
                return

            print(f"✅ 최종 파일 경로: {self.selected_file}")
            self.data = self.selected_file  # 데이터 저장

    def run_search(self, keyword):
        search_word = quote(keyword)
        file=make_CSV(search_word)
        if file:
            QMessageBox.information(self, '완료', f'채용정보가 {file}로 저장되었습니다!')
    
    def generate_company_info(self):
        """ 채용공고 CSV를 기반으로 기업 정보 크롤링 """
        output_file = crawl_company_info(self.selected_file)  # 기업정보 크롤링 실행
        if output_file:
            QMessageBox.information(self, '완료', f'기업정보가 {output_file}로 저장되었습니다!')
    
    def visualize_data(self):
        DataFrame1=pd.read_csv(self.selected_file1)
        DataFrame2=pd.read_csv(self.selected_file2)    
        print(DataFrame2['사원수'])    
        DataFrame=merge_type_cast(DataFrame1,DataFrame2)
        DF=top_rank_sort(DataFrame1,DataFrame2,10,'평균연봉')
        print(DF['사원수'])
        DF1=top_rank_sort(DataFrame1,DataFrame2,10,'매출액(2023)')
        print(DF1['사원수'])
        DF2=top_rank_sort(DataFrame1,DataFrame2,10,'영업이익(2023)')
        print(DF2['사원수'])
        vizual_DF_bar(DF,DF1,DF2,sort=['평균연봉','매출액(2023)','영업이익(2023)'])
        vizual_DF_pie(DF1)
    def extract_preference_keywords(self):
        if self.selected_file:
            subprocess.run(["python", "Make_word_Text_.py", self.selected_file])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RecruitmentApp()
    window.show()
    sys.exit(app.exec())
