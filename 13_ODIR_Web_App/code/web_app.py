"""
ODIR (Ocular Disease Intelligent Recognition) 웹 애플리케이션
- CSV 데이터 분석 및 DB 업로드
- 이미지 분석 및 결과 표시
- 웹 인터페이스 제공
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import json
import torch
import torchvision.transforms as transforms
from sqlalchemy import create_engine

from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

# 경로 설정 - 프로젝트 루트 기준 상대 경로
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train_images')
TEST_DIR = os.path.join(DATA_DIR, 'test_images')
CSV_PATH = os.path.join(DATA_DIR, 'full_df.csv')
MODEL_DIR = os.path.join(DATA_DIR, 'models')
UPLOAD_DIR = os.path.join(DATA_DIR, 'uploads')
TEMPLATES_DIR = os.path.join(ROOT_DIR, 'templates')
STATIC_DIR = os.path.join(ROOT_DIR, 'static')

# 폴더 생성
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 데이터베이스 설정
DB_URL = 'mysql+mysqlconnector://2:1234@192.168.2.154:3306/g6'  # DB 접속 정보
DB_TABLE_NAME = 'odir_data'  # 기본 테이블명

# 질병 정보
DISEASE_NAMES = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other']
DISEASE_COLS = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

# Flask 앱 초기화
app = Flask(__name__, 
            template_folder=TEMPLATES_DIR,
            static_folder=STATIC_DIR)
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 제한

# 허용된 파일 확장자
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """허용된 파일 확장자인지 확인"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_model():
    """모델 로드 함수"""
    # import가 이 함수 내에서만 필요하므로 지역 import
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from odir_analysis import load_model_for_inference
    
    model_path = os.path.join(MODEL_DIR, 'odir_model.pth')
    print(f"모델 경로 확인: {model_path}")
    if os.path.exists(model_path):
        try:
            print("모델 로드 시도 중...")
            model = load_model_for_inference(model_path)
            if model is not None:
                print("모델 로드 성공!")
            else:
                print("모델 로드 실패: None 값 반환")
            return model
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return None
    else:
        print(f"모델 파일을 찾을 수 없음: {model_path}")
        print(f"디렉토리 내용: {os.listdir(os.path.dirname(model_path)) if os.path.exists(os.path.dirname(model_path)) else '(디렉토리 없음)'}")
        return None

def analyze_image(image_path):
    """이미지 분석 함수"""
    # 새로운 예측 모듈 사용
    from predict import predict_image
    
    try:
        # 모델 로드
        model = get_model()
        if model is None:
            print("모델을 로드할 수 없습니다")
            return {"error": "모델을 로드할 수 없습니다"}
        
        print(f"이미지 경로: {image_path}")
        # 이미지 예측 - 새로운 함수 사용
        results = predict_image(model, image_path, device="cpu")
        
        if all(value == 0.0 for value in results.values()):
            print("경고: 모든 예측 값이 0입니다. 예측에 문제가 있을 수 있습니다.")
        
        return results
    
    except Exception as e:
        print(f"이미지 분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def upload_csv_to_db(csv_path=CSV_PATH, table_name=DB_TABLE_NAME, if_exists_option='replace'):
    """데이터베이스에 CSV 파일 업로드"""
    try:
        # MySQL 연결
        engine = create_engine(DB_URL, echo=False)
        
        # CSV 파일 존재 확인
        if not os.path.exists(csv_path):
            print(f"CSV 파일을 찾을 수 없음: {csv_path}")
            return False
        
        # CSV 파일 로드
        df = pd.read_csv(csv_path)
        print(f"CSV 파일 로드 성공: {len(df)} 레코드")
        
        # 이미지 경로 수정 (상대경로 -> 절대경로)
        def convert_path(file_path):
            if pd.isna(file_path):
                return None
                
            file_name = os.path.basename(file_path)
            
            # 훈련 이미지 디렉토리에서 확인
            train_path = os.path.join(TRAIN_DIR, file_name)
            if os.path.exists(train_path):
                return train_path
                
            # 테스트 이미지 디렉토리에서 확인
            test_path = os.path.join(TEST_DIR, file_name)
            if os.path.exists(test_path):
                return test_path
                
            return file_path  # 찾지 못한 경우 원래 경로 반환
        
        # 경로 컬럼이 있는 경우 절대 경로로 변환
        if 'filepath' in df.columns:
            df['filepath'] = df['filepath'].apply(convert_path)
            
        if 'filename' in df.columns:
            df['abs_filepath'] = df['filename'].apply(convert_path)
        
        # 기본 업로드 필드가 없는 경우
        if 'filepath' not in df.columns and 'filename' not in df.columns:
            print("경로 관련 컬럼이 없어 이미지 경로를 추가하지 않았습니다.")
        
        # 데이터프레임을 DB에 저장
        df.to_sql(table_name, engine, if_exists=if_exists_option, index=False)
        
        if if_exists_option == 'append':
            print(f"DB의 {table_name} 테이블에 {len(df)} 레코드를 추가했습니다.")
        else:
            print(f"DB의 {table_name} 테이블을 {len(df)} 레코드로 교체했습니다.")
        return True
        
    except Exception as e:
        print(f"CSV 파일 업로드 중 오류 발생: {e}")
        return False

def get_data_from_db(table_name=DB_TABLE_NAME, limit=1000):
    """데이터베이스에서 데이터 가져오기"""
    try:
        # MySQL 연결
        engine = create_engine(DB_URL, echo=False)
        
        # 데이터 조회
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        df = pd.read_sql(query, engine)
        
        print(f"DB에서 {len(df)} 레코드를 가져왔습니다.")
        return df
    
    except Exception as e:
        print(f"DB 연결 중 오류 발생: {e}")
        return None

# Flask 라우트
@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """파일 업로드 페이지"""
    if request.method == 'POST':
        # 파일 확인
        if 'file' not in request.files:
            return jsonify({'error': '파일이 선택되지 않았습니다'}), 400
            
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': '파일이 선택되지 않았습니다'}), 400
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # 이미지 분석
            results = analyze_image(filepath)
            
            # 결과 반환
            if 'error' in results:
                return jsonify({'error': results['error']}), 500
                
            # 결과 페이지로 리다이렉트
            return redirect(url_for('result', filename=filename))
        
        return jsonify({'error': '허용되지 않는 파일 형식입니다'}), 400
        
    return render_template('upload.html')

@app.route('/result/<filename>')
def result(filename):
    """분석 결과 페이지"""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # 파일 존재 확인
    if not os.path.exists(filepath):
        return jsonify({'error': '파일을 찾을 수 없습니다'}), 404
        
    # 이미지 분석
    results = analyze_image(filepath)
    
    if 'error' in results:
        return jsonify({'error': results['error']}), 500
        
    # 결과 페이지 렌더링
    return render_template('result.html', 
                          filename=filename, 
                          results=results, 
                          disease_names=DISEASE_NAMES)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """업로드된 파일 서빙"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """이미지 분석 API"""
    try:
        # 업로드된 이미지 가져오기
        if 'file' not in request.files:
            return jsonify({'error': '이미지 파일이 없습니다'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '파일이 선택되지 않았습니다'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': '허용되지 않는 파일 형식입니다'}), 400
            
        # 파일 저장
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 이미지 분석
        results = analyze_image(filepath)
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': f'분석 중 오류 발생: {str(e)}'}), 500

@app.route('/api/db_stats')
def api_db_stats():
    """데이터베이스 통계 정보 API"""
    try:
        # DB 연결
        engine = create_engine(DB_URL, echo=False)
        
        # 통계 정보 조회
        query = f"SELECT COUNT(*) as total FROM {DB_TABLE_NAME}"
        total_count = pd.read_sql(query, engine).iloc[0]['total']
        
        # 질병별 통계 (DISEASE_COLS 기준으로 컬럼이 있는 경우)
        disease_stats = {}
        for disease, col in zip(DISEASE_NAMES, DISEASE_COLS):
            try:
                query = f"SELECT COUNT(*) as count FROM {DB_TABLE_NAME} WHERE {col} = 1"
                disease_count = pd.read_sql(query, engine).iloc[0]['count']
                disease_stats[disease] = int(disease_count)
            except:
                disease_stats[disease] = 0
        
        stats = {
            'total_records': int(total_count),
            'disease_stats': disease_stats
        }
        
        return jsonify(stats)
    
    except Exception as e:
        return jsonify({'error': f'통계 조회 중 오류 발생: {str(e)}'}), 500

@app.route('/api/upload_csv', methods=['POST'])
def api_upload_csv():
    """CSV 파일을 DB에 업로드하는 API"""
    try:
        # 파일 확인
        if 'file' not in request.files:
            return jsonify({'error': 'CSV 파일이 없습니다'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '파일이 선택되지 않았습니다'}), 400
            
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'CSV 파일만 업로드 가능합니다'}), 400
            
        # 파일 저장
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(temp_path)
        
        # 테이블 이름과 업로드 방식 가져오기
        table_name = request.form.get('table_name', DB_TABLE_NAME)
        if_exists = request.form.get('if_exists', 'replace')
        
        # DB에 업로드
        success = upload_csv_to_db(temp_path, table_name, if_exists)
        
        if success:
            return jsonify({'message': f'{file.filename} 파일이 DB에 성공적으로 업로드되었습니다'})
        else:
            return jsonify({'error': 'DB 업로드 중 오류가 발생했습니다'}), 500
            
    except Exception as e:
        return jsonify({'error': f'파일 업로드 중 오류 발생: {str(e)}'}), 500

@app.route('/db_admin')
def db_admin():
    """데이터베이스 관리 페이지"""
    return render_template('db_admin.html')

@app.route('/view_data')
def view_data():
    """데이터 조회 페이지"""
    try:
        # 데이터베이스에서 데이터 가져오기
        limit = request.args.get('limit', 100, type=int)
        table_name = request.args.get('table', DB_TABLE_NAME)
        
        df = get_data_from_db(table_name, limit)
        
        if df is None or len(df) == 0:
            return render_template('view_data.html', error='데이터를 가져올 수 없습니다')
            
        # 데이터프레임을 HTML 테이블로 변환
        html_table = df.to_html(classes='data-table', index=False, border=0)
        
        return render_template('view_data.html', table=html_table, count=len(df))
        
    except Exception as e:
        return render_template('view_data.html', error=f'오류 발생: {str(e)}')

@app.route('/stats')
def stats():
    """통계 정보 페이지"""
    try:
        # DB 연결
        engine = create_engine(DB_URL, echo=False)
        
        # 통계 정보 조회
        query = f"SELECT COUNT(*) as total FROM {DB_TABLE_NAME}"
        total_count = pd.read_sql(query, engine).iloc[0]['total']
        
        # 질병별 통계
        disease_stats = {}
        for disease, col in zip(DISEASE_NAMES, DISEASE_COLS):
            try:
                query = f"SELECT COUNT(*) as count FROM {DB_TABLE_NAME} WHERE {col} = 1"
                disease_count = pd.read_sql(query, engine).iloc[0]['count']
                disease_stats[disease] = int(disease_count)
            except:
                disease_stats[disease] = 0
        
        return render_template('stats.html', 
                              total=int(total_count), 
                              disease_stats=disease_stats,
                              disease_names=DISEASE_NAMES)
        
    except Exception as e:
        return render_template('stats.html', error=f'오류 발생: {str(e)}')

# 메인 실행 코드
if __name__ == '__main__':
    # 시작 시 CSV 파일을 DB에 업로드 (자동화)
    if os.path.exists(CSV_PATH):
        print(f"CSV 파일({CSV_PATH})을 DB에 업로드 중...")
        upload_csv_to_db(CSV_PATH)
    else:
        print(f"CSV 파일을 찾을 수 없음: {CSV_PATH}")
    
    # Flask 앱 실행
    app.run(host='0.0.0.0', port=5000, debug=True)
