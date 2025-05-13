from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import torch
import json
import torchvision.transforms as transforms
import os
#http://localhost:5000/
app = Flask(__name__, static_folder='static', template_folder='templates')

# 모델과 클래스 이름 로드
MODEL_PATH = 'bird_model_web.pt'
CLASS_NAMES_PATH = 'bird_class_names.json'

# 이미지 전처리 함수
def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)

# 예측 함수
def predict(image_bytes):
    # 모델과 클래스 이름 로드
    model = torch.jit.load(MODEL_PATH)
    model.eval()
    
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)
    
    # 이미지 전처리 및 예측
    tensor = preprocess_image(image_bytes)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()
    
    # 결과 반환
    result = {
        'predicted_class': class_names[predicted_idx],
        'confidence': float(probabilities[predicted_idx]) * 100,
        'predictions': {class_names[i]: float(prob) * 100 for i, prob in enumerate(probabilities)}
    }
    
    return result

# 메인 페이지 라우트
@app.route('/')
def index():
    return render_template('index.html')

# 예측 API 엔드포인트
@app.route('/predict', methods=['POST'])
def predict_api():
    if 'image' not in request.files:
        return jsonify({'error': '이미지가 없습니다.'}), 400
    
    image_file = request.files['image']
    image_bytes = image_file.read()
    
    try:
        result = predict(image_bytes)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)