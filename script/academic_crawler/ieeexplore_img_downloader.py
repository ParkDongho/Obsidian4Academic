import os
import json
import requests
import argparse
from selenium import webdriver
import time

def download_images_with_selenium(data_file, download_dir, driver):
    """
    셀레니움과 쿠키를 사용하여 이미지 다운로드 함수

    Args:
        data_file (str): JSON 파일 경로
        download_dir (str): 이미지 다운로드 디렉토리 경로
        driver (webdriver): Selenium WebDriver 인스턴스
    """
    # 디렉토리 생성
    os.makedirs(download_dir, exist_ok=True)

    # JSON 데이터 로드
    with open(data_file, 'r') as f:
        data = json.load(f)

    # Selenium에서 쿠키 가져오기
    cookies = driver.get_cookies()
    session = requests.Session()

    # requests 세션에 쿠키 추가
    for cookie in cookies:
        session.cookies.set(cookie['name'], cookie['value'])

    # 이미지 다운로드
    for item in data:
        image_url = item.get('image_href')
        file_name = item.get('img_file_name')

        if image_url and file_name:
            file_path = os.path.join(download_dir, file_name)
            try:
                response = session.get(image_url, stream=True)
                if response.status_code == 200:
                    with open(file_path, 'wb') as img_file:
                        for chunk in response.iter_content(1024):
                            img_file.write(chunk)
                    print(f"Downloaded: {file_path}")
                else:
                    print(f"Failed to download {image_url}, status code: {response.status_code}")
            except Exception as e:
                print(f"Error downloading {image_url}: {e}")

if __name__ == "__main__":
    # 아규먼트 파서 설정
    parser = argparse.ArgumentParser(description="Download images using Selenium and cookies.")
    parser.add_argument('data_file', type=str, help="Path to the JSON file containing image data.")
    parser.add_argument('download_dir', type=str, help="Directory to save the downloaded images.")
    parser.add_argument('url', type=str, help="URL to the authenticated page.")
    args = parser.parse_args()

    # Selenium WebDriver 설정
    driver = webdriver.Chrome()  # 필요에 따라 WebDriver 경로 지정
    time.sleep(3)
    print(args.url)
    driver.get(args.url)
    time.sleep(3)

    # 사용자 입력 대기 (로그인 등 필요 시)
    input("Complete any required actions in the browser, then press Enter...")

    # 이미지 다운로드 함수 실행
    download_images_with_selenium(args.data_file, args.download_dir, driver)

    # WebDriver 종료
    driver.quit()
