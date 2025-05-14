 # 요청 보내기
# url = "https://www.vworld.kr/po_geolocation.do"
# params = {
#     "lon": lon,
#     "lat": lat
# }
# response = requests.get(url, params=params)

# # 결과 보기
# print("응답 결과:", response.text)

# 위치 정보
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pyautogui
import time
import pandas as pd
# ==================== 설정 ====================
locationDF=pd.read_csv(r'C:\Users\Administrator\Desktop\DL\DATA\midpoints.csv')
locationDF.fillna('교차로',inplace=True)
# ==================== 셀레니움 드라이버 실행 ====================
driver=webdriver.Chrome(service=Service(ChromeDriverManager().install()))
# for x in range(len(location.shapep[0]))
driver.get("https://map.vworld.kr/map/dtkmap.do?mode=MAPD101")
driver.maximize_window()
wait=WebDriverWait(driver, 20)
time.sleep(2)

# 분석
wait.until(EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), '건물분석')]"))).click()
time.sleep(2)
wait.until(EC.element_to_be_clickable((By.XPATH, "//p[contains(text(), '일조권 분석')]"))).click()
time.sleep(2)
count=0
sunDF=pd.DataFrame()
for x in range(locationDF.shape[0]):
  total_sun_list=[]
  max_sun_list=[]
  for y in [[1,2],[3,4],[5,6]]:
    # 1,2: 시작점 3,4: 중간점, 5,6: 끝점점
    lon=locationDF.iloc[x,y[1]]
    print(lon)
    lat=locationDF.iloc[x,y[0]]
    print(lat)
    wait.until(EC.element_to_be_clickable((By.ID, "selectPoint_bldg03"))).click()
    time.sleep(4)
    # FlyTo 
    flyto_script = f"""
    const lon = {lon};
    const lat = {lat};
    const height = 200;
    const destination = Cesium.Cartesian3.fromDegrees(lon, lat, height);
    map3d._wsViewer.camera.flyTo({{
      destination: destination,
      duration: 2
    }});
    """
    driver.execute_script(flyto_script)
    time.sleep(5)

    # ==================== 비율 기반 클릭 위치 계산 ====================
    screen_w, screen_h = pyautogui.size()
    ratio_x=0.5125
    ratio_y=0.5319
    click_x=int(screen_w * ratio_x)
    click_y=int(screen_h * ratio_y)
    # print(f"({click_x}, {click_y})")
    # 수동클릭
    time.sleep(2)
    pyautogui.moveTo(click_x, click_y)
    pyautogui.click()

    # 분석시작작
    time.sleep(4)
    driver.find_element(By.ID, "bldg03-finishSet").click()
    time.sleep(10)
    # 결과과
    wait.until(lambda d: d.find_element(By.ID, "continuetime").text.strip() != "")
    total_sun = driver.find_element(By.ID, "continuetime").text
    max_sun = driver.find_element(By.ID, "alltime").text
    print("총 일조량:", total_sun)
    print("연속 일조량:", max_sun)
    total_sun_list.append(total_sun)
    max_sun_list.append(max_sun)
    time.sleep(6)
    wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="t3dbldg-result-pop"]/div[1]/button[2]'))).click()
    time.sleep(2)
  a=pd.DataFrame([[locationDF.iloc[x,0],locationDF.iloc[x,1],locationDF.iloc[x,2],total_sun_list[0],max_sun_list[0],
                locationDF.iloc[x,3],locationDF.iloc[x,4],total_sun_list[1],max_sun_list[1],
                locationDF.iloc[x,5],locationDF.iloc[x,6],total_sun_list[2],max_sun_list[2]]])
  sunDF=pd.concat([sunDF,a])
  count+=1
  print(count)
sunDF.to_csv(f'sun_location({count}).csv',index=False)
driver.quit() 