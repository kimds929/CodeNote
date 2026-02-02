# 파이썬 크롤링 기초
# activate env01
import os
import getpass

import urllib
import urllib.request as req
from urllib.error import URLError, HTTPError
import requests
#import lxml
import lxml.html

import numpy as np
import pandas as pd


# urllib 사용법 및 기본 스크랩핑 ----------------------------------------------
# 파일 URL
    # 이미지 주소
img_url = 'https://ichef.bbci.co.uk/images/ic/720x405/p05x7423.jpg'

#다운받을경로
username = getpass.getuser()    # 사용자명 얻기
Desktop_adr =  'C:\\Users' + '\\'+ username + '\Desktop'    # Desktop 주소

save_path = Desktop_adr + '\\test1.jpg'    # 이미지파일 다운경로


# (urlretrieve) 해당 url파일을 문서에 저장하는 방법
    # file, header = request.urlretrieve(URL,'저장위치'): 'URL'에 해당하는 파일을 '저장위치'로 내보내겟다.
    # file : 저장 위치 (해당경로 저장 이행)
    # header : header정보를 return
img_file, header = req.urlretrieve(img_url, save_path)   
print(header)  # header정보 추출

# (urlopen) 해당 url파일을 python program으로 가져와 처리하는 방법
    # improt urllib.request as req : 함수 호출
    # req.urlopen(URL) : URL을 Open
    # req.urlopen(URL).read() : Open한 URL파일을 읽는 실행문


# --- urllib Package관련 기본 함수 -------
response = req.urlopen(img_url)
contents = response.read()  # response된 데이터를 읽기

print(response.info())    # header정보 추출
response.getheaders()   # header정보를 list형태로 호출

response.getcode()      # states Code를 Return (response.status ) ※ 200번이 정상
    # response.status          # status Code 호출 ( = mem.getcode() )

response.geturl()        # 수신된 Adress를 호출

# ------------------------------------------


# 발생할 수 있는 Error들 : HTTPError, URLError

# 파일쓰기
with open(save_path, 'wb') as c:    # save_path경로를 open
    c.write(contents)       # URL에서 불러온 파일을 쓰기




# lxml 사용 기초 스크래핑-----------------------------------------------------------------------
    # 네이버 메일 뉴스 사이트 url 스크레이핑하기
response = requests.get("https://www.naver.com")    # 스크래핑 대상을 get방식으로 호출
    # GET 방식과 POST 방식을 구분하여 추출할줄 알아야함

urls = []   # url 지정할 list 생성
names = []  # 신문사명 지정할 list 생성
root = lxml.html.fromstring(response.content)   # html source를 호출

# response.content(=requests.get("Adress").content) 해당페이지 소스보기 했을때 나오는 HTML Source
for a in root.cssselect('.api_list .api_item a.api_link'):  # naver 메인 뉴스기사 사이트명에 대한 css 주소
    url = a.get('href')                         # 'href'속성내 항목을 추출하여 url변수에 저장
    print( url ) 
    urls.append(url)  

    name = a.cssselect('img')[0].get('alt')     # 하위 img 태그 'alt'속성내 항목을 추출하여 url변수에 저장
    print( name ) 
    names.append(name)  

pd.DataFrame(np.array([names, urls]).T, columns=['name','link'])




# xpath 사용 기초 스크래핑-----------------------------------------------------------------------
# 세션정보를 활용하기
session = requests.Session()    # 세션 접속 (접속내용을 기억하고 있는다.)
response = session.get("https://www.naver.com")    # 스크래핑 대상을 get방식으로 호출
root = lxml.html.fromstring(response.content)   # html source를 호출

# session.close()     # 세션종료
urls = {}   # url 지정할 Dictonary 형태로 생성
root = lxml.html.fromstring(response.content)   # html source를 호출

# response.content(=requests.get("Adress").content) 해당페이지 소스보기 했을때 나오는 HTML Source
for a in root.xpath('//ul[@class="api_list"]/li[@class="api_item"]/a[@class="api_link"]'):  # naver 메인 뉴스기사 사이트명에 대한 xpath 주소
    # // : 현재경로, / : 하위경로, Tag[@class="Class명"]
    
    #print(lxml.html.tostring(a, pretty_print=True))     # 해당부분 HTML 문서구조를 보여줌
    name = a.xpath('./img')[0].get('alt')   # 하위 img 태그 'alt'속성내 항목을 추출하여 url변수에 저장
    print( name )  

    url = a.get('href')
    print( url )

    urls[name] = url    # Dictonary[Key] = value → {key : value}

print(urls)




# CSS selector 실습 --------------------------------------------------------------------
import lxml.html

tree = lxml.html.fromstring('''<!DOCTYPE html>
<html>
<head>
    <title>lxml tutorials</title>
</head>
<div>
    <div class="cc cv"><i>Hello</i> <i>World!!!</i></div>
    <div class="cc"><p id="abc">Hello lxml</p></div>
    <div class="cc"><a href="http://adress.com">lxml Study</a></div>
</div>
</html>''')

selectors = tree.cssselect('.cc')

len(selectors)  # 해당 css 선택자 갯수

# 선택자 단어추출 text_content()
selectors[0].cssselect('i')[0].text_content()  # Hello
selectors[0].cssselect('i')[1].text_content()  # World!!!

selectors[1].text_content()  # Hello lxml
selectors[1].cssselect('p')[0].text_content()  # Hello lxml

selectors[2].text_content()  # lxml Study
selectors[2].cssselect('a')[0].text_content()  # lxml Study

# 선택자 속성 추출
selectors[0].attrib  # {'class': 'cc cv'} # 속성확인
selectors[0].get('class')  # cc cv   # 'class'값 홗인

selectors[1].cssselect('p')[0].attrib  # {'id': 'abc'} 속성확인
selectors[1].cssselect('p')[0].get('id')  # 'abc   # 'id'값 홗인

selectors[2].cssselect('a')[0].attrib  # {'href': 'http://adress.com'} #  속성확인
selectors[2].cssselect('a')[0].get('href')  # http://adress.com   # 'href' 속성값 얻기

#--------------------------------------------------------------------------------------------






# GET 방식 데이터 통신 -----------------------------------------------------

# 기본요청1 -----
url = "http://www.encar.com/"

response = urllib.request.urlopen(url)

    # url 주소 파싱 (GET방식의 이해) : 해당 query가 공개되기 떄문에 주로 게시판 형식에서 사용
urllib.parse.urlparse("http://www.encar.com/?id=test&pw=1111")    # url주소를 파싱
urllib.parse.urlparse("http://www.encar.com/??id=test&pw=1111").query    # 파싱한 url주소중 query부분만 도출


# 기본요청2 -----
API = 'https://api.ipify.org'

# GET방식 parameter
values ={'format':'json'}       # json 대신 text나 jsonp로 변경하면 출력형식이 달라짐
params = urllib.parse.urlencode(values) # 'format=json'   # Dictonary 형태를 Query 형태로 변환
params

ip = requests.get('https://api.ipify.org').text
print('My public IP address is: {}'.format(ip))

# 요청 URL 생성
url = API + '?' + params
url

# 수신 데이터 읽기
data = urllib.request.urlopen(url).read()

# 수신 데이터 디코딩
text = data.decode('UTF-8')
text


# RSS 데이터 가져오기 --- # 행정안전부 사이트(https://www.mois.go.kr/) RSS 데이터 수신 실습 
    # RSS : 사이트에서 보내주는 소식지(XML형태)
API = "https://www.mois.go.kr/gpms/view/jsp/rss/rss.jsp"
    # ?ctxCd=1001
params = []
site_code =  [1001, 1012, 1013, 1014]

for num in site_code:
    params.append(dict(ctxCd=num))

params

# 연속해서 4회 요청
for c in params:
    # URL 인코딩
    param = urllib.parse.urlencode(c)   # {key : value} → key=value   # Dictonary 형태를 Query 형태로 변환
     # print(param)
    
    # URL 출력
    url = API + "?" + param
    # print(url)

    # 요청
    res_data = urllib.request.urlopen(url).read()   # 해당 url 연결 및 read
    contents = res_data.decode('UTF-8')    # 수신 후 디코딩
    print(contents)





# --------- Fake-UserAgent 사용 / Header정보(referer) 삽입  (개발자 도구 활용) ------------
    # 파이썬 크롤링 실습 - Daum 증권정보 가져오기
    # 다음주식 site : http: // finance.daum.net/

# Fake-Header정보 (가상으로 UserAgent 생성)
ua = UserAgent()
# print(ua.ie)  # 익스플로러
# print(ua.msie)  # 익스플로러
# print(ua.chrome)  # 크롬
# print(ua.safari)  # 사파리 (아이폰, 아이패드)
# print(ua.random)    # 랜덤

# 헤더정보
headers = {
    'user-agent': ua.ie,
    'referer': 'http://finance.daum.net/'
}

# 다음 주식정보 요청 URL
url = 'http://finance.daum.net/api/search/ranks?limit=10'   # 주식정보 요청 주소

# user-agent정보와 referer정보를 header에 전달하여 request.urlopen 실시
response = req.urlopen(req.Request(url, headers=headers))
response_read = response.read().decode('utf-8')

# 응답데이터 확인(Json Data)
print("res : ", response_read)

# 응답데이터 str → json변환 및 data값 출력
rank_json = json.loads(response_read)['data']        # 'data' : key값

pd.DataFrame(rank_json)  # DataFrame형태로 추출

for info in rank_json:
    print("순위 : {}, 금액 : {}, 회사명 {}" .format(
        info["rank"], info["tradePrice"], info["name"]))




# --------- Cookie 정보 활용 --------------------------------------------------------
session = requests.Session()    # 세션활성화    
    # session.close()      # 세션 종료(비활성화)

# 쿠키 리턴     # https://httpbin.org/ 실습용 가짜 Data를 얻을 수 있는 사이트
request1 = session.get("https://httpbin.org/cookies", cookies={'name':'kim1'})     # 스크래핑 대상을 get방식으로 호출
request1.text    # 보낸 쿠키정보 확인

# 쿠키 set (서버쪽 정확한 접근)
request2 = session.get("https://httpbin.org/cookies/set", cookies={'name':'kim2'})     # 스크래핑 대상을 get방식으로 호출
request2.text    # 보낸 쿠키정보 확인

request1.status_code     # 수신상태 코드 확인 (200 : 정상)
    #request.ok      # 수신상태 Boolen으로 확인


# User-Agent
url = "https://httpbin.org/"
headers = {'user-agent' : 'nice-man_1.0.0_win10_ram16_home_chrome'}

# header 정보 전송
request3 = session.get(url, headers = headers) # , cookies = {'name':'kim3'})
    # with문 사용 권장 → 파일, DB, HTTP통신 등  



# ------- httpbin 사이트를 이용한 JSON 실습 ------------------------------------
# JSON : 데이터 교환형식, 경량데이터, XML대체, 특정언어에 종속되지 않음
    # https://jsonplaceholder.typicode.com : JSON Data 실습을 위한 가짜 데이터를 내려주는곳
    # https://httpbin.org/ 실습용 가짜 Data를 얻을 수 있는 사이트

s = requests.session()  # 세션 열기
r = s.get("https://httpbin.org/stream/100", stream = True )       
# JSON형식 Data100개 가져오기,  stream = True : 데이터를 직렬화해서 가져옴
print(r.headers)    # header정보
r.headers['Content-Type']   # 컨텐츠 타입 : JSON
print(r.text)       # Json정보 출력
print(r.encoding)   # 인코딩 정보


if r.encoding is None:      # 인코딩 정보 없을경우 부여
    r.encoding = 'UTF-8'

print(r.encoding)   # 인코딩 정보


json_list = []      # list형식내 Dictonary형식을 부여하여 DataFrame형태로 반환하기 위한 빈 List생성
for line in r.iter_lines(decode_unicode=True):  # decode_unicode=True : 문자깨짐 방지를 위함
    # print(type(line))       # Data_Type : string
    # print(line)
    
    # JSON 딕셔너리 형태로 변환후 타입 확인
    b = json.loads(line)    # String을 Dictonary 형태로 변환
    # print(type(b))       # Data_Type : Dictonary
    # print(b)

    for k, v in b.items():
        print('key : {} / value : {}' .format(k, v))

    json_list.append(b)     # JSON Data List내 Dictonary 형태로 만들어줌
    print()
    
df_json = pd.DataFrame(json_list)   # JSON데이터를 List → DataFrame형태로 변환
df_json.info()      # DataFrame 정보

s.close()


# -- Rest API : GET, POST, DELETE, PUT: UPDATE, REPALCE(FETCH: UPDATE, MODIFY) ---------
    # 중요 : URL을 활용하여 자원의 상태정보를 주고받는 모든것을 의미
s = requests.session()
r = s.get("https://api.github.com/events")

# r.status_code
r.raise_for_status()        # 호출에 에러가 있으면 에러 발생, 에러가 없어야 다음줄 코딩 실행가능
# r.headers
r.headers['Content-Type']   # 컨텐츠 타입 : JSON

jar = requests.cookies.RequestsCookieJar()  # 쿠키설정
jar.set('name', 'good-man', domain ="httpbin.org", path='/cookies') #쿠키삽입


    # GET 방식
r = s.get("https://httpbin.org/cookies", cookies = jar, timeout = 5) # 요청, timeout : 수신을 일정시간만큼 기다림
r.text

    # POST방식
payload =  {"id":"test77", "pw":"111"}
r = s.post("http://httpbin.org/post", data = payload, cookies=jar )
r.text

    # PUT방식
r = s.put("http://httpbin.org/put", data = payload)
r.text

    # PUT방식
r = s.delete("https://jsonplaceholder.typicode.com/posts/1")
r.text

s.close()

