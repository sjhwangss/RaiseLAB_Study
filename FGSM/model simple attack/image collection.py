import os
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time


def download_images(search_query, num_images, save_path):
    # ChromeDriver 경로 설정
    driver = webdriver.Chrome()  # Chromedriver의 실행 파일 경로를 지정해야 합니다.

    # Google Images 접속
    driver.get("https://www.google.com/imghp")
    search_box = driver.find_element(By.NAME, "q")
    search_box.send_keys(search_query)
    search_box.send_keys(Keys.RETURN)

    # 이미지 URL 수집
    image_urls = set()
    while len(image_urls) < num_images:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        thumbnails = driver.find_elements(By.CSS_SELECTOR, "img.Q4LuWd")

        for img in thumbnails:
            try:
                img.click()
                time.sleep(1)
                large_image = driver.find_element(By.CSS_SELECTOR, "img.n3VNCb")
                src = large_image.get_attribute("src")
                if src and "http" in src:
                    image_urls.add(src)
                if len(image_urls) >= num_images:
                    break
            except Exception:
                pass

    # 다운로드 디렉토리 생성
    os.makedirs(save_path, exist_ok=True)

    # 이미지 다운로드
    for i, url in enumerate(image_urls):
        try:
            response = requests.get(url)
            with open(os.path.join(save_path, f"{search_query}_{i}.jpg"), "wb") as file:
                file.write(response.content)
        except Exception:
            pass

    driver.quit()
    print(f"Downloaded {len(image_urls)} images for {search_query}")


# 실행: 각각 다른 폴더에 저장
download_images("cat", 50000, "cats")  # cat 이미지를 cats 폴더에 저장
download_images("dog", 50000, "dogs")  # dog 이미지를 dogs 폴더에 저장
