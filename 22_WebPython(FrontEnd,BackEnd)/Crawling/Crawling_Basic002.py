# activate env01
import os
import getpass

import urllib
import urllib.request as req
import urllib.parse as ps
import requests     # 데이터 통신 모듈
import json
import fake_useragent
from fake_useragent import UserAgent
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

import numpy as np
import pandas as pd
import time



# BeautifulSoup 사용 스크래핑 ----------------------------------------------------
    # html 예제 활용
html ="""
<html>
    <head>
        <title>The Dormouse's story</title>
    </head>
    <body>
        <h1>this is h1 area</h1>
        <h2>this is h2 area</h2>
        <p class="title"><b>The Dormouse's story</b></p>
        <p class="story">Once upon a time theare were three little sites.
            <a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>
            <a href="http://example.com/lacie" class="sister" id="link2">Lacie</a>
            <a data-io="link3" href="http://example.com/little" class="brother" id="linke3">Title</a>
        </p>
        <p>
            <p class="story">
                story....
            </p>
        </p>
    </body>
</html>
"""

soup = BeautifulSoup(html, 'html.parser')      # bs4 초기화
    # html : requests.get("url").text   OR  ur.urlopen("url").read()
    # Parser(구문분석기) : html.parser, lxml, lxml-xml, xml, html5lib

soup.prettify()  # 보기 좋게 출력 (자동 들여쓰기)

soup.html.body.h1    # h1 태그에 접근
soup.html.body.p     # 여러개의 p태그: html > body안의 가장 첫번째 p태그 내용을 가져옴

soup.html.body.p.next_sibling.next_sibling
soup.html.body.p.next_sibling.next_sibling.next_sibling.next_sibling

    # 텍스트출력
soup.html.body.h1.text
soup.html.body.h1.get_text()
soup.html.body.h1.string

    # 다음 엘리먼트 확인
soup.html.body.p.next_element  # Next태그(하위태그 포함)로 접근
soup.html.body.p.next_sibling.next_sibling.next_element



# (태그로 접근) find, find_all --------------------------------------------------------------------------
soup.find('p')                  # p태그중 가장 먼저 위치한 내용을 추출
soup.find(class_='sister')      # sister 클래스중 가장 먼저 위치한 내용을 추출

soup.find_all('a', limit=2)           # 태그에 해당하는 모든 데이터 추출,  limit = 가져올 태그갯수
soup.find_all('a', class_='sister')   # 태그와 class를 모두 만족하는 태그정보를 추출
soup.find_all('a', id='link2')        # 태그와 ID를 모두 만족하는 태그정보를 추출
soup.find_all('a', string='Elsie')    # 태그와 string을 모두 만족하는 태그정보를 추출
soup.find_all('a', string=['Title'])  # 태그와 string을 모두 만족하는 태그정보를 추출

    # 속성내 특정조건을 지칭할때
soup.find_all('a', {'class':'brother', 'data-io':'link3'})    # Dictonary형태에 속성값을 사용자 편의로 지정가능
soup.find('a', {'class':'brother', 'data-io':'link3'})        # Dictonary형태에 속성값을 사용자 편의로 지정가능


# (CSS선택자로 접근) select_one, select --------------------------------------------------------------------------
soup.select_one('p.title >b')                # CSS선택자 접근법을 활용가능
soup.select_one('a[data-io="link3"]')        # []를 통해 속성값에도 접근 가능
soup.select('p.story > a:nth-of-type(2)')    # a태그내에 2번째 열을 추출




# 네이버 이미지 다운로드 예제 ------------------------------------------------
    #다운받을경로
username = getpass.getuser()    # 사용자명 얻기
Desktop_adr = 'C:\\Users' + '\\'+ username + '\Desktop'    # Desktop 주소
save_folder = Desktop_adr + '\\img_down'

if not (os.path.isdir(save_folder) ):
    os.makedirs(save_folder)

    # 이미지경로
base_url = 'https://search.naver.com/search.naver?where=image&sm=tab_jum&query='
quote = urllib.parse.quote_plus('포스코')   # 검색어를 parsing 해서 형태를 변환
url = base_url + quote
url

    # header정보 초기화
opener = urllib.request.build_opener()
ua = UserAgent()
opener.addheaders = [('user-agent', ua.ie)]     # 헤더정보 삽입
urllib.request.install_opener(opener)
response = req.urlopen(url)

    # header정보 초기화2 (같은방법)
headers = {'user-agent': ua.ie, 'referer': url}
# user-agent정보와 referer정보를 header에 전달하여 request.urlopen 실시
response = req.urlopen(req.Request(url, headers=headers))

    # BeautifulSoup를 활용하여 해당 사이트 HTML Document 호출
soup = BeautifulSoup(response, "html.parser")
soup.prettify()

img_list = soup.select('div > a.thumb._thumb > img')
# img_list[0]['data-source']        # data-source속성값 가져오기
# img_list[0].attrs['data-source']  # data-source속성값 가져오기2
# img_list[0].get('data-source')    # data-source속성값 가져오기3

for i, img_url in enumerate(img_list):
    filename = save_folder + '\\' + str(i) + '.png'
    urllib.request.urlretrieve(img_url['data-source'] , filename)



# 로그인 처리 --------------------------------------------------------------------------------
    # form Data
log_info = {
    "redirectUrl": "http://www.danawa.com/",
    "loginMemberType": "general",
    "id": "ID",
    "password": "PASSWORD"
    }

    # header 정보
headers = {
    'user-agent': UserAgent().chrome,
    'referer': 'https://auth.danawa.com/login?url=http%3A%2F%2Fwww.danawa.com%2F'
}

with requests.session() as s:
        # Request : 로그인 시도
    res = s.post("https://auth.danawa.com/login", log_info, headers = headers)    # 로그인 페이지에서 정보를 넘겨줌

    if res.status_code != 200:
        raise Exception("Login_failed!")

    # res.content.decode('UTF-8')   # 내용 확인

        # 로그인 후에 세션정보를 가지고 페이지 이동
    res = s.get("https://buyer.danawa.com/order/Order/orderList", headers = headers)
    # res.encdoing = 'euc-kr'    # 한글이 깨질경우

        # bs4 초기화
    soup = BeautifulSoup(res.text, "html.parser")

        # 로그인 성공 체크
    check_name = soup.select("div.nav_top > p.user > strong")
    check_name

    if check_name is None:
        raise Exception("Login_failed!")



# (Web 자동화) Selenium 사용 --------------------------------------------------------------------------
from selenium import webdriver

browser = webdriver.Chrome('./chrome/chromedriver.exe')     # Web Driver 실행파일 경로 설정
browser.implicitly_wait(5)  # 크롬브라우저 내부 대기

# dir(browser)    # 사용가능한 함수 확인

browser.set_window_size(1920, 1280)     # 브라우저 크기 설정
    # maximize_window() : 최대사이즈 윈도우로 실행,   minimize_window() : 최소사이즈 윈도우로 실행

browser.get("https://www.daum.net")     # 페이지 이동
browser.page_source     # 페이지 html-source 가져오기
browser.session_id      # 세션값 출력
browser.title           # 타이틀 출력
browser.current_url     # 접근한 Page의 url
browser.get_cookies()   # 쿠키정보출력

    # 가져올 정보 추출
element = browser.find_element_by_css_selector('#q')    # 검색창 input-box선택
element.send_keys('포스코')      # 키를 입력
# element.submit()                    # Enter Key입력

search_btn =  browser.find_element_by_css_selector('div > button.ico_pctop.btn_search') # 버튼위치 선택
search_btn.click()      # Search_button 클릭


    #다운받을경로
username = getpass.getuser()    # 사용자명 얻기
Desktop_adr = 'C:\\Users' + '\\'+ username + '\Desktop'    # Desktop 주소

    # 스크린샷 저장
browser.save_screenshot(Desktop_adr + "\\web1.png")         # 해당경로에 스크린샷 저장
browser.get_screenshot_as_file(Desktop_adr + "\\web2.png")  # 해당경로에 스크린샷 저장2

print("complete!")
browser.quit()  #브라우저 종료




#  Selenium 사용 실습 --------------------------------------------------------------
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

chrome_opttions = Options()     # 옵션실행
chrome_opttions.add_argument("--headless")      # selenium 실행시 창이 뜨지 않게 함

    # webdriver 설정 - Headless모드
browser = webdriver.Chrome('./chrome/chromedriver.exe', options=chrome_opttions)     # Web Driver 실행파일 경로 설정(창비활성화 모드)
    # options=chrome_opttions  :  창 비활성화

browser.implicitly_wait(5)  # 크롬브라우저 내부 대기
browser.maximize_window()     # 브라우저 크기 설정
    # maximize_window() : 최대사이즈 윈도우로 실행,   minimize_window() : 최소사이즈 윈도우로 실행

    # 크롤링할 페이지로 이동
browser.get("http://prod.danawa.com/list/?cate=112758&15main_11_02")
# browser.page_source    # 해당 페이지 html-source보기


    #(Explicitly wait) 제조사정보 더보기 클릭하기 (CSS-selector)
plus_btn_css = 'dd > div.spec_opt_view > button.btn_spec_view.btn_view_more'
WebDriverWait(browser, 3).until(EC.presence_of_element_located((By.CSS_SELECTOR, plus_btn_css))).click()
    # 3초간 기다리되 plus_button의 모든 요소가 제자리에 위치하면 실행

    #(Explicitly wait) 제조사정보 더보기 클릭하기 (Xpath)
# plus_btn_xpath = '//*[@id="dlMaker_simple"]/dd/div[2]/button[1]'
# WebDriverWait(browser, 3).until(EC.presence_of_element_located((By.XPATH, plus_btn_xpath))).click()
    # 3초간 기다리되 plus_button의 모든 요소가 제자리에 위치하면 실행

    # 제조사정보 더보기 클릭하기(CSS-selector : WebDriverWait 사용하지 않고 실시하기)
# time.sleep(3)       # 무조건 3초간 정지
# browser.find_element_by_css_selector(plus_btn_css).click()   # 버튼위치 선택

browser.close()










