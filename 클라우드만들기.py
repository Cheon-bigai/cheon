from konlpy.tag import Okt
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd


text = open('hiring.txt',encoding='utf-8').read() #텍스트 파일 넣기
okt=Okt()

okt_morphs = okt.nouns(text)
print(okt_morphs)
print(type(okt_morphs))

choose = [i for i,word in enumerate(okt_morphs) if word =='우대'] #우대  리스트 순서 추출 / 넣고 싶은 단어 넣으세요

prefer =[]

for i in choose:                                    #우대 뒤부터 단어 10개
    prefer.extend(okt_morphs[i +1 : i+11]) # 몇개 뒤까지 설정

print(f'우대 뒤 단어 10개 {prefer}' )
 
word_counts = Counter(prefer)

df = pd.DataFrame(word_counts.items(), columns=['단어', '빈도'])
df = df.sort_values(by='빈도', ascending=False).reset_index(drop=True)
print(df)


img_mask = np.array(Image.open('cloud.png'))

wordcloud = WordCloud(
    font_path='malgun.ttf',  # 한글 폰트 지정 (윈도우: 'malgun.ttf', 맥: 'AppleGothic')
    background_color='white',
    width=800,
    height=400,
    mask=img_mask
).generate_from_frequencies(word_counts)

# 워드클라우드 표시
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()