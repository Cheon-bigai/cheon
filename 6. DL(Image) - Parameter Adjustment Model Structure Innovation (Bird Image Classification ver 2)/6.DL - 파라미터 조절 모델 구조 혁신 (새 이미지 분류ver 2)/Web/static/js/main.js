document.addEventListener('DOMContentLoaded', function() {
  const uploadForm = document.getElementById('upload-form');
  const imageUpload = document.getElementById('image-upload');
  const previewCard = document.getElementById('preview-card');
  const previewImage = document.getElementById('preview-image');
  const resultCard = document.getElementById('result-card');
  const resultClass = document.getElementById('result-class');
  const resultConfidence = document.getElementById('result-confidence');
  const probabilityBars = document.getElementById('probability-bars');
  const predictBtn = document.getElementById('predict-btn');

  // 이미지 미리보기
  imageUpload.addEventListener('change', function(e) {
      if (e.target.files.length > 0) {
          const file = e.target.files[0];
          const reader = new FileReader();
          
          reader.onload = function(e) {
              previewImage.src = e.target.result;
              previewCard.style.display = 'block';
              resultCard.style.display = 'none';
          };
          
          reader.readAsDataURL(file);
      }
  });

  // 폼 제출 처리
  uploadForm.addEventListener('submit', async function(e) {
      e.preventDefault();
      
      if (!imageUpload.files.length) {
          alert('이미지를 선택해주세요.');
          return;
      }
      
      const formData = new FormData();
      formData.append('image', imageUpload.files[0]);
      
      // 버튼 로딩 상태
      predictBtn.disabled = true;
      predictBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> 분석 중...';
      
      try {
          const response = await fetch('/predict', {
              method: 'POST',
              body: formData
          });
          
          const data = await response.json();
          
          if (response.ok) {
              displayResults(data);
          } else {
              alert('오류: ' + (data.error || '알 수 없는 오류가 발생했습니다.'));
          }
      } catch (error) {
          alert('요청 처리 중 오류가 발생했습니다: ' + error.message);
      } finally {
          // 버튼 상태 복원
          predictBtn.disabled = false;
          predictBtn.innerHTML = '분석하기';
      }
  });

  // 결과 표시 함수
  function displayResults(data) {
      // 예측 클래스 및 신뢰도 표시
      resultClass.textContent = '예측 결과: ' + data.predicted_class;
      resultConfidence.textContent = '신뢰도: ' + data.confidence.toFixed(2) + '%';
      
      // 클래스별 확률 막대 그래프 생성
      probabilityBars.innerHTML = '';
      
      // 확률 내림차순으로 정렬
      const sortedPredictions = Object.entries(data.predictions)
          .sort((a, b) => b[1] - a[1]);
      
      // 확률 막대 그래프 생성
      sortedPredictions.forEach(([className, probability]) => {
          const progressDiv = document.createElement('div');
          progressDiv.className = 'progress';
          
          const progressBar = document.createElement('div');
          progressBar.className = 'progress-bar';
          progressBar.style.width = probability + '%';
          progressBar.textContent = className + ': ' + probability.toFixed(2) + '%';
          
          // 최고 확률 클래스 강조
          if (className === data.predicted_class) {
              progressBar.classList.add('bg-success');
          }
          
          progressDiv.appendChild(progressBar);
          probabilityBars.appendChild(progressDiv);
      });
      
      // 결과 카드 표시
      resultCard.style.display = 'block';
  }
});