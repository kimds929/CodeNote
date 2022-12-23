# 네이버 아파트 시세 1
# activate env01
import pandas as pd
import os
import getpass

import urllib.request as req
from urllib.error import URLError, HTTPError
import requests
#import lxml
import lxml.html

u = r"https://search.naver.com/search.naver?sm=tab_sly.hst&where=nexearch&query="
r = r"&oquery="
l = r"&tqi=UM74PlpVuElssbnxCXKsssssthw-498671&acr=1"
search = '가락 쌍용1차'

url = u + search + r +search + l

response = requests.get(url) 
root = lxml.html.fromstring(response.content)   # html source를 호출
# 'td:nth-child(1)'  #거래구분
# 'td:nth-child(4).fs' # 평형
# 'td:nth-child(5).fs' # 가격
# 'td:nth-child(6).fs' # 층수


column_names = ['DealGroup','PyeongArea','Price','floor']
css_codes = ['td:nth-child(1)','td:nth-child(4).fs', 'td:nth-child(5).fs', 'td:nth-child(6).fs']

apt_sise = pd.DataFrame()
for i, k in enumerate(css_codes):
    all_contents = [] 
    for c in root.cssselect(k): 
        content = c.text_content()    # 내용을 호출
        if k == 'td:nth-child(1)':
            if content in ('매매','전세','월세'):
                all_contents.append(content)
        else:
            all_contents.append(content)
    apt_sise[i] = all_contents

apt_sise.columns = column_names
apt_sise.filter("DealGroup"==['매매'])
apt_sise.filter(like='매매', axis=0)


apt_sise
apt_sise[(apt_sise["PyeongArea"]=='81/59') &(apt_sise["DealGroup"]=='매매') ]
len(apt_sise)










