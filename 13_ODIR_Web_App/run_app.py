"""
ODIR 웹 애플리케이션 실행 스크립트
"""

import os
import sys

# 코드 폴더 추가
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code'))

# 웹 앱 모듈 import
from web_app import app, upload_csv_to_db, CSV_PATH

def main():
    """메인 실행 함수"""
    print("===== ODIR 안구질환 분석 웹 애플리케이션 시작 =====")
    
    # CSV 파일 DB 자동 업로드 여부 확인
    if len(sys.argv) > 1 and sys.argv[1] == '--upload-csv':
        if os.path.exists(CSV_PATH):
            print(f"CSV 파일({CSV_PATH})을 DB에 업로드 중...")
            upload_csv_to_db(CSV_PATH)
        else:
            print(f"CSV 파일을 찾을 수 없음: {CSV_PATH}")
    
    # 웹 애플리케이션 실행
    host = '0.0.0.0'  # 모든 IP 주소에서 접근 가능
    port = 5000  # 기본 포트
    
    print(f"웹 서버 시작: http://{host}:{port}")
    print("확인하려면 웹 브라우저에서 http://localhost:5000 에 접속하세요")
    print("종료하려면 Ctrl+C를 누르세요")
    
    # Flask 앱 실행
    app.run(host=host, port=port, debug=True)

if __name__ == "__main__":
    main()
