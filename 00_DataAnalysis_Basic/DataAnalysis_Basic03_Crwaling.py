from bs4 import BeautifulSoup

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
        <div class='content'>
            <span>
                <h3>contents</h3>
                <span> this book is written by Joshep </span>
            </span>
            <ul>
                <li>chapter1. intro</li>
                <li>chapter2. story</li>
                <li>chapter3. appendix</li>
            </ul>
        </div>
        <p class="content">Once upon a time theare were three little sites.
            <a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>
            <a href="http://example.com/lacie" class="sister" id="link2">Lacie</a>
            <a data-io="link3" href="http://example.com/little" class="brother" id="linke3">Title</a>
        </p>
        <p>
            <p class="content">
                story....
            </p>
        </p>
    </body>
</html>
"""





soup = BeautifulSoup(html, 'html.parser')      # bs4 초기화
    # html : requests.get("url").text   OR  ur.urlopen("url").read()
    # Parser(구문분석기) : html.parser, lxml, lxml-xml, xml, html5lib

print(soup.prettify())  # 보기 좋게 출력 (자동 들여쓰기)

soup.html.body.h1    # h1 태그에 접근
soup.html.body.p     # 여러개의 p태그: html > body안의 가장 첫번째 p태그 내용을 가져옴

# 옆 element 추출 (sibling)
soup.html.body.p
soup.html.body.p.next_sibling
soup.html.body.p.next_sibling.next_sibling
soup.html.body.p.next_sibling.next_sibling.next_sibling
soup.html.body.p.next_sibling.next_sibling.next_sibling.next_sibling

    # 텍스트출력
soup.html.body.h1.text
soup.html.body.h1.get_text()
soup.html.body.h1.string

    # 다음 element 확인
soup.html.body.p.next_element  # Next태그(하위태그 포함)로 접근
soup.html.body.p.next_sibling.next_sibling.next_element

    # 하위 element 추출 (child)
soup.span
soup.span.find_parent()

soup.span.findChild()

soup.span.findNext()
soup.span.findNext().findNext()




# (태그로 접근) find, find_all --------------------------------------------------------------------------
soup.find('p')                  # p태그중 가장 먼저 위치한 내용을 추출
soup.find('ul').find('li')
soup.find(class_='sister')      # sister 클래스중 가장 먼저 위치한 내용을 추출

soup.find_all('a')
soup.find_all('a')[2]
soup.find('ul')
soup.find('ul').find_all('li')

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




# 웹 페이지에서 직접 가져오기
import requests

url = 'https://finance.naver.com/'
response = requests.get(url)
response

html = response.text
soup = BeautifulSoup(html)      # bs4 초기화
soup.find_all('tbody', id='_topItems1')[0].find_all('a')

top_items_src = soup.find_all('tbody', id='_topItems1')[0].find_all('a')


top_items = []
for i in top_items_src:
    # break
    top_items.append(i.text)

i
i.find_parent()
i.find_parent().find_next_sibling()
i.find_parent().find_next_sibling().text


top_items = []
for i in top_items_src:
    item = i.text
    value = i.find_parent().find_next_sibling().text
    top_items.append([item, value])

import pandas as pd
pd.DataFrame(top_items, columns=['종목','시세'])