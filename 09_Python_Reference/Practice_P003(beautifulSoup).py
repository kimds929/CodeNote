from pandas import Series, DataFrame
import pandas as pd
import numpy as np
from numpy import nan as NA
import copy

#import win32clipboard
import pyperclip        #Clipboard 관련 Package

        #---- XML 처리
import requests
import xml.etree.ElementTree as et
import bs4
from bs4 import BeautifulSoup as bs
import lxml
import urllib.request as ur


# ------ 가공 함수 --------
def remove_all(x,y):        # list에서 항목을 모두 지우는 함수
    x = list( filter(lambda a: a!= y, x) )
    return x

def xml_depth(soup):            # XML문서의 Max Depth를 알려주는 함수
    if hasattr(soup, "contents") and soup.contents:
        return max([xml_depth(child) for child in soup.contents ])+1
    else:
        return 0

# 마지막 node인 beautifulsoup list 파일을 DataFrame으로 바꿔주는 함수
def soup_pd(soupdtl, tag_list) :        # soupdt = 마지막 node인 beautifulsoup list,  tag_list =  tag값
    l1 = []
    l2 = []
    for i in range(0, len(soupdtl)):
        l2 = []
        for j in range(0, len(tag_list)):
            k = soupdtl[i].find_all(tag_list[j])[0]
            l2.append(k.text)
        l1.append(l2)
    df = pd.DataFrame(l1)
    df.columns = tag_list
    return df


URL = 'http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp?stnId=108'
    #'http://effbot.org/zone/element-index.htm'

    #------------ ??????
response = requests.get(URL)
status = response.status_code
get = requests.get(URL)

text = response.text        # Response된 내용을 보는 명령어
print(text)


et.fromstring(response.text)        # 오류??
et.parse(response.text)

    # ------------ xml.etree.ElementTree    Package
request =ur.urlopen(URL)
xml = request.read()
xml_et = et.fromstring(xml)
xmlfile = et.ElementTree(response.text)
print(xmlfile)

    # --------- urllib.request / BeautifulSoup   Pacakge
request =ur.urlopen(URL)
xml = request.read()
#soup = bs(xml, "html.parser")
soup = bs(xml, "lxml-xmlr")
soup = bs(xml, "xml")
soupbd = soup.body

soupdt  = soupbd.find("data")      # find : 값에 해당하는 맨앞 데이터 1개만 추출
soupdtl  = soupbd.find_all("data")      # find : 값에 해당하는 전체 데이터 추출

print( soupdt.prettify() )   # HTML문서를 보기 좋게
pyperclip.copy(soup.prettify())     # Clipboard 에 복사하기
pyperclip.paste()                   # Clipboard 값 붙여넣기
print( soup.get_text() )    # Tag내 Text값 가져오기


soupdt.contents     # XML 내용(Contents)을 List 형식으로 Return
soupdt.children     # XML 내용(Contents)을 List_iterator 형식으로 Return → list()를 넣어주면 List로 변환가능
soupdt.descendants      # XML의 하위 항목을 generator 형식으로 모두 Return → list()를 넣어주면 List로 변환가능
remove_all( soupdt.contents, "\n")

xml_depth(soupdt)       # (Function) Depth Return

list( soupdt.parents )     # Parents: 상위 항목을 도출
list( soupdt.children )    # Chile: 하위 항목을 도출
list( soupdt.descendants ) # Descendants : 하위 항목을 모두도출

souploc  = soup.find("location")
remove_all( list( souplchild1[3].children ), "\n")
    #-----------------------------------------------------

xml_depth(soupdt)
    # 마지막 nod 에서 Data추출
l1 = []
for i in soupdt.find_all():
    print(l1.append(i.text))


soupdt.find_all() == soupdtl[0].find_all()

# 마지막 nod에서 DataFrame 형태로 추출 * soupdtl = 마지막 Nod (Depth=2)의 집합 List
len(soupdtl)
soupdtl[0:2]
xml_depth(soupdt)
    # 전체 Data의경우
l1 = []
l2 = []
for i in range(0,len(soupdtl)):
    l2 = []
    for j in soupdtl[i].find_all() :
        l2 .append(j.text)
    l1.append(l2)

df1 = pd.DataFrame(l1)   # list → DataFrame
df1.head(10)
df1.to_clipboard(sep = ' ')        #Clipboard로 내보내기

    # 특성 tag만 추출하는경우
c = ['mode','tmef','wf']
l1 = []
l2 = []
for i in range(0,len(soupdtl)):
    l2 = []
    for j in soupdtl[i].find_all(c) :
        l2 .append(j.text)
    l1.append(l2)
df1 = pd.DataFrame(l1)
df1.columns = c

df1 = pd.DataFrame(l1)  # list → DataFrame
df1.head(10)
df1.to_clipboard(sep = ' ')        #Clipboard로 내보내기


#---------------------------------------------  추가연습
import urllib.request as ur
from bs4 import BeautifulSoup as bs
import lxml



xmlStr = '''
            <users>
                    <user grade="gold">
                        <name>Kim Cheol Soo</name>
                        <age>25</age>
                        <birthday>19940215</birthday>
                        <성별>남</성별>
                    </user>
                    <user grade="diamond">
                        <name>Kim Yoo Mee</name>
                        <age>21</age>
                        <birthday>19980417</birthday>
                        <성별>여</성별>
                    </user>
            </users>
        '''
s1 = bs(xmlStr, "html.parser")
s2 = bs(xmlStr, "lxml-xml")
s3 = bs(xmlStr, "xml")
s1.find("user")


s2_user=  s2.find_all("user")
s2_user
c = ['name','성별','birthday','age']


soup_pd(s2_user, c)


# XML 설명 블로그 : https://blog.naver.com/jwyoon25/221352211611
# XML 설명 블로그2 : https://codetravel.tistory.com/22
# XML 설명 공식Page : https://docs.python.org/2/library/xml.etree.elementtree.html

