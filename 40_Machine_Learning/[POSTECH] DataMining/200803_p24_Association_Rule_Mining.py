
# import apyori
# import surprise

#!/usr/bin/env python
# coding: utf-8


# import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from apyori import apriori


import sys
sys.path.append('d:\\Python\\★★Python_POSTECH_AI\\DS_Module')    # 모듈 경로 추가
from DS_DataFrame import *
from DS_OLS import *

absolute_path = 'D:/Python/★★Python_POSTECH_AI/Dataset_AI/DataMining/'

# 연관규칙 찾기위한 지표계산
# · 지지도(support, 전체 거래 항목중 A제품과 제품B을 동시에 구매하는 규칙) :  A,B가 동시에 구매되는 거래의 수 / 전체거래의 수 
    # 전체 거래항목 중 상품 A와 상품 B를 동시에 포함하여 거래하는 비율을 의미하는데요,
    # A -> B 라고 하는 규칙이 전체 거래 중 차지하는 비율을 통해 해당 연관 규칙이 얼마나 의미가 있는 규칙인지를 확인

# · 신뢰도(Confidence, A제품을 사면 B제품을 사는 규칙) : A,B가 동시에 구매되는 거래의 수 / A가 구매되는 거래의 수 
    # 상품 A를 포함하는 거래 중 A와 B가 동시에 거래되는 비중으로,
    # 상품 A를 구매 했을 때 상품 B를 구매할 확률이 어느정도 되는지를 확인

# · 향상도(Lift) : 상품 A의 거래 중 항목 B가 포함된 거래의 비율 / 전체 상품 거래 중 상품 B가 거래된 비율
    # P(A∩B) / P(A)*P(B)  = P (B|A) / P (B)
    # 품목 A와 B사이에 아무런 관계가 상호 관계가 없으면 향상도는 1이고,
    # 향상도가 1보다 높아질 수록 연관성이 높다고 할 수 있습니다. (B만 구매하는경우보다, A와B가 함꼐 구매되는 경우보다 많다)
    # 이것은 또한 향상도가 1보다 크거나 작다면 우연적 기회(random chance)보다 우수하다고도 해석


# create example data for association rule mining
    # List of List 형태로 변형을 해주어야 함
trans = [['b', 'c', 'g'],
        ['a', 'b', 'd', 'e', 'f'],
        ['a', 'b', 'c', 'g'],
        ['b', 'c', 'e', 'f'],
        ['b', 'c', 'e', 'f', 'g']]
print(trans)


# parameter setting
    # Search 범위를 잘 설정해야 함, 너무 낮게 잡으면 데이터셋이 클경우 잘 안돌아감
min_supp = 0.5      # Support
min_conf = 0.6      # Confidence
min_lift = 1.01     # Lift


# extract association rules
# ?apriori
rules = apriori(transactions=trans, min_support=min_supp, min_confidence=min_conf, min_lift=min_lift)
results = list(rules)

print(results[0])
print(results[0])
# RelationRecord(items=frozenset({'g', 'c'}), support=0.6,  
#                   # c를 샀을때 g를 살 확률
#             ordered_statistics=[OrderedStatistic(items_base=frozenset({'c'}), 
#             items_add=frozenset({'g'}), 
#             confidence=0.7499999999999999, 
#             lift=1.2499999999999998),
#                   # g를 샀을때 c를 살 확률
#             OrderedStatistic(items_base=frozenset({'g'}),
#             items_add=frozenset({'c'}),
#             confidence=1.0, lift=1.25)])

# define print function
def apriori_results(results, printing=True, return_df=True):
    if return_df:
        result = pd.DataFrame()
    
    for line in results:
        for i in range(len(line.ordered_statistics)):
            a = line.ordered_statistics[i].items_base
            b = line.ordered_statistics[i].items_add
            supp = line.support
            conf = line.ordered_statistics[i].confidence
            lift = line.ordered_statistics[i].lift
            if printing:
                print(a, '->', b, '| support: %0.3f' % supp, '| confidence: %0.3f' % conf, '| lift: %0.3f' % lift)
            if return_df:
                result_part_series = pd.Series()
                result_part_series['A'] = a
                result_part_series['B'] = b
                result_part_series['support'] = round(supp,3)
                result_part_series['confidence'] = round(conf,3)
                result_part_series['lift'] = round(lift,3)
                result = pd.concat([result, pd.DataFrame([result_part_series])], axis=0)

    if return_df:
        return result
    else:
        return None

# print the result
apriori_results(results, printing=False)


c = pd.DataFrame()
a= pd.Series()
a['a'] = 1
a['b'] = 2
pd.concat([c,a],axis=1)













#!/usr/bin/env python
# coding: utf-8

# import libraries
import numpy as np
import pandas as pd


# 1. read instacart dataset
df = pd.read_csv(absolute_path + 'order_products__train.csv', nrows=10000)  # nrows=10000 : 1만개만 불러와라
df.head()
df_info = DS_DF_Summary(df)


df.shape
# df.T.apply(lambda x: x.to_numpy()[:3],axis=1)
# df.T.apply(lambda x: x.head(2).to_numpy().tolist() + x.tail(2).to_numpy().tolist(),axis=1)
# df.T.apply(lambda x: x.unique(),axis=1)

df_gb = df.groupby(by='order_id').apply(lambda x: x['product_id'].to_numpy().tolist())
df_gb

df.groupby(by='order_id').apply(lambda x: x['product_id'].to_numpy().tolist()).to_frame()

trans_product = df_gb.to_numpy().tolist()

    # apriori
rules_product = apriori(transactions=trans_product,
                    min_support=0.01, min_confidence=0.01, min_lift=0)
results_product = list(rules_product)
results_product

product_results = apriori_results(results_product, printing=False)
product_results.describe()

a = pd.Series()
a['abc'] = 1
a['bcd'] = 2
pd.DataFrame([a])




# # 2. read microsoft dataset   ** ???????????
# f = open('anonymous-msweb.data', 'r')

# data = f.readlines()
# data[:10]
# data[1000:1010]



# customers = []
# items = []
# for row in data:
#     row_list = row.split(sep=',')
#     if row_list[0] == 'C':
#         row_customer = row_list[1]
#     if row_list[0] == 'V':
#         customers.append(row_customer)
#         items.append(row_list[1])
# df2 = pd.DataFrame(np.c_[customers, items], columns=['users', 'items'])
# df2.head()


# trans2 = []

# a2 = df2.groupby(['users']).groups
# for id in a2.keys():
#     trans2.append(df2.iloc[a2[id]]['items'].values)

# for i in range(5):
#     print(trans2[i])





# 3. read random shopping cart dataset
df3 = pd.read_csv(absolute_path + 'random_shopping_cart.csv', names=['date', 'id', 'item'])
df3

df3_gb = df3.groupby(by='id').apply(lambda x: x['item'].to_numpy().tolist())
df3_gb

trans3 = df3_gb.to_numpy().tolist()
for i in range(5):
    print(trans3[i])


unq_item, cnt_item = np.unique(df3['item'], return_counts=True)
plt.hist(cnt_item)


    # apriori
rules_gb = apriori(transactions=trans3, 
                    min_support=0.1, min_confidence=0.1, min_lift=0)
                    # min_support=0.01, min_confidence=0.01, min_lift=0)
results_gb = list(rules_gb)
results_gb


gb_results = apriori_results(results_gb, printing=False)
gb_results
gb_results.describe()