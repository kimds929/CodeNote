# https://s3.ap-northeast-2.amazonaws.com/teamlab-gachon/json_example.json  # 데이터다운로드
import json
import os

# 예제 데이터
# {"employees":[
#     {"firstName":"John", "lastName":"Doe"},
#     {"firstName":"Anna", "lastName":"Smith"},
#     {"firstName":"Peter", "lastName":"Jones"}
# ]}

# JSON 파일 읽기
with open('Database\json_example.json', 'r', encoding='utf8') as f:
    contents=f.read()
    json_data = json.loads(contents)
    print(json_data)

json_data
json_data['employees'][0]


# JSON 파일 쓰기 
    # Dict Type으로 데이터 저장 → JSON 모듈로 Write
dict_data = {'Name':'Zara', 'Age':7, 'Class':'First'}

with open('Database\json_example.json', 'w') as f:
    json.dump(dict_data, f)


# csv로 저장된 ipa110106.xml파일을 JSON으로 변환
    # https://s3.ap-northeast-2.amazonaws.com/teamlab-gachon/ipa110106.xml
    # https://jjeongil.tistory.com/201
    # http://blog.naver.com/PostView.nhn?blogId=pk3152&logNo=221367256441
import xmltodict
os.listdir('Database')

with open('Database\ipa110106.xml', 'r', encoding='utf8') as f:
    xmlString=f.read()
    print(xmlString)
    xmlToJson = xmltodict.parse(xmlString)
    print(xmlToJson)

xmlToJson['item']
xmlToJson['item']['rnum']
xmlToJson['item']['bldNm']

json_type = json.dumps(xmlToJson, indent=4)
print(json_type)
dict2_type = json.loads(json_type)
dict2_type

