import pandas as pd
import numpy as np
import scipy as sp
import math

from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
from sklearn import datasets

# del(customr_df)       #변수 삭제
df = pd.read_clipboard()  #Clipboard로 입력하기
# df.to_clipboard()        #Clipboard로 내보내기
df = pd.read_csv('Database/supermarket_sales.csv')

# sklearn Dataset Load : iris, wine, breast_cancer
def Fun_LoadData(datasetName):
    from sklearn import datasets
    load_data = eval('datasets.load_' + datasetName + '()')
    data = pd.DataFrame(load_data['data'], columns=load_data['feature_names'])
    target = pd.DataFrame(load_data['target'], columns=['Target'])
    df = pd.concat([target, data], axis=1)
    for i in range(0, len(load_data.target_names)):
        df.at[df[df['Target'] == i].index, 'Target'] = str(load_data.target_names[i])   # 특정값 치환
    return df

    # breast_cancer Dataset
df = Fun_LoadData('breast_cancer')
df.info()
df.head()

pd.set_option('display.float_format', '{:.3f}'.format) # 항상 float 형식으로
pd.set_option('display.float_format', '{:.2e}'.format) # 항상 사이언티픽
pd.set_option('display.float_format', '${:.2g}'.format)  # 적당히 알아서
pd.set_option('display.float_format', None) #지정한 표기법을 원래 상태로 돌리기: None

# 【 연관규칙(Association Rule) 】 --------------------------------------------------------------------------------------------------------
    # 제품 판매 데이터를 분석하여 연관관계가 있음을 찾아내는 기법


dataMarket1 = [['오렌지쥬스', '콜라'], ['우유','오렌지쥬스','세제'],['오렌지쥬스','비누'],['오렌지쥬스','비누','콜라'],['세제','콜라']]
df_Market1 = pd.DataFrame(dataMarket1)
dataMarket2=[['사과','치즈','생수'], ['생수','호두','치즈','고등어'], ['수박','사과','생수'], ['생수','호두','치즈','옥수수']]

# 동시발생 테이블(Co-occurrence table) 작성

# 연관규칙 찾기위한 지표계산
# · 지지도(support, 전체 거래 항목중 A제품과 제품B을 동시에 구매하는 규칙) :  A,B가 동시에 구매되는 거래의 수 / 전체거래의 수 
    # 전체 거래항목 중 상품 A와 상품 B를 동시에 포함하여 거래하는 비율을 의미하는데요,
    # A -> B 라고 하는 규칙이 전체 거래 중 차지하는 비율을 통해 해당 연관 규칙이 얼마나 의미가 있는 규칙인지를 확인

# · 신뢰도(Confidence, A제품을 사면 B제품을 사는 규칙) : A,B가 동시에 구매되는 거래의 수 / A가 구매되는 거래의 수 
    # 상품 A를 포함하는 거래 중 A와 B가 동시에 거래되는 비중으로,
    # 상품 A를 구매 했을 때 상품 B를 구매할 확률이 어느정도 되는지를 확인

# · 향상도(Lift, Improvement) : 상품 A의 거래 중 항목 B가 포함된 거래의 비율 / 전체 상품 거래 중 상품 B가 거래된 비율
    # P(A∩B) / P(A)*P(B)  = P (B|A) / P (B)
    # 품목 A와 B사이에 아무런 관계가 상호 관계가 없으면 향상도는 1이고,
    # 향상도가 1보다 높아질 수록 연관성이 높다고 할 수 있습니다. (B만 구매하는경우보다, A와B가 함꼐 구매되는 경우보다 많다)
    # 이것은 또한 향상도가 1보다 크거나 작다면 우연적 기회(random chance)보다 우수하다고도 해석


# 데이터 전처리
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

basket_lst = []
for sublist in df_Market1.values.tolist():
    clean_subliist = [item for item in sublist if item is not None and item is not np.nan]
    basket_lst.append(clean_subliist)

basket_lst

te = TransactionEncoder()
te_ary = te.fit(basket_lst).transform(basket_lst)

df_basket = pd.DataFrame(te_ary, columns = te.columns_)
df_basket


# Apriori 알고리즘 적용
freq_itemsets = apriori(df_basket, min_support = 0.3, use_colnames = True)   # 규칙찾기
freq_itemsets


# association_rules을 활용하여 연관규칙 찾기
from mlxtend.frequent_patterns import association_rules
association_rules(freq_itemsets, metric = 'confidence', min_threshold = 0.7)    # 연관규칙찾기: confidence 값이 0.7 이상인것
association_rules(freq_itemsets, metric = 'support', min_threshold = 0.4)       # 연관규칙찾기: support 값이 0.4 이상인것
rules = association_rules(freq_itemsets, metric = 'lift', min_threshold = 2)    # 연관규칙찾기: lift 값이 1.0 이상인것
rules

rules['antecedent_len'] = rules['antecedents'].apply(lambda x: len(x))      # 제품갯수를 지정하는 column 추가
rules

rules[ (rules['antecedent_len'] >= 2) & rules['confidence'] > 0.75 ]
rules[ rules['antecedents'] == {'우유'} ]
rules[ rules['antecedents'] == {'우유', '오렌지쥬스'} ]




# ---------------------------------------------------------------------------------------------------------------------------