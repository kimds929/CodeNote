import pandas as pd
df = pd.read_csv('Database/supermarket_sales.csv')
df = pd.read_clipboard()  #Clipboard로 입력하기

df.describe()
df.info()
df.head()
df['City'].drop_duplicates().tolist()  # pandas Series to list

# ---- Test Data ------------------------------------------------------------------
test_list = [['a',1,'A',20], ['b',5,'B',10], ['c',3,'B',30], ['d',9,'A',60], ['e',7,'B',40]]
test_df = pd.DataFrame(test_list, columns=['A','B','C','D'])

test_df2 = test_df.set_index(['A','B','C'])
test_df2
# ----------------------------------------------------------------------------------------


import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt

# https://jjangjjong.tistory.com/9
Array1= [33, 16, 13, 38, 45, 24, 29, 17, 22, 13, 9, 8, 16, 32, 34, 5, 20, 3, 20, 20, 6, 33, 46, 27, 12, 29, 18, 33, 41, 3]
Array2= [50, 26, 45, 3, 1, 44, 45, 34, 49, 4, 17, 35, 5, 12, 16, 2, 21, 6, 28, 36, 22, 16, 5, 47, 30, 48, 37, 45, 38, 37]
Array3= [4, 38, 35, 49, 37, 8, 24, 26, 44, 26, 29, 4, 36, 35, 25, 48, 12, 22, 7, 5, 12, 18, 20, 31, 33, 19, 29, 33, 34, 23]
Array4= ['A', 'A', 'A', 'B', 'A', 'A', 'A', 'B', 'A', 'A', 'A', 'A', 'B', 'A', 'B', 'A', 'A', 'B', 'B', 'A', 'B', 'A', 'A', 'A', 'A', 'B', 'A', 'A', 'B', 'A']
Array5= ['A', 'B', 'C', 'B', 'A', 'A', 'A', 'C', 'C', 'C', 'C', 'A', 'C', 'B', 'C', 'A', 'C', 'B', 'A', 'B', 'C', 'A', 'B', 'A', 'C', 'B', 'A', 'C', 'A', 'A']
Array6= ['A', 'B', 'E', 'D', 'E', 'A', 'B', 'D', 'A', 'D', 'A', 'B', 'D', 'D', 'A', 'D', 'A', 'C', 'E', 'D', 'B', 'B', 'D', 'C', 'A', 'A', 'E', 'E', 'C', 'D']

# random_list = []
# for i in range(0,30):
#     if random.random()>=0.5:
#         random_list.append('A')
#     else:
#         random_list.append('B')
# random_list

df = pd.DataFrame([Array1, Array2, Array3, Array4, Array5, Array6]).T
df.columns = ['n1', 'n2', 'n3', 'g1', 'g2', 'g3']
df['n1'] = df['n1'].astype('int')
df['n2'] = df['n2'].astype('int')
df['n3'] = df['n3'].astype('int')

df.info()
df.head()

df[df['g3'].isin(['A','B'])==False]


df_group = df_final.groupby(['출강목표','YP_상한'])
df_group_c = groupAnalysis_df.copy()


# ----------------------------------------------------------------------------------------
def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_


df_Calc = df_group[['초_YP', '누적SPM_EL']].agg(['mean', 'std',  percentile(10), percentile(90)])

pd.concat([df_group_c, df_Calc], axis=1).to_clipboard()

df_origin_group = df_result.groupby(['출강목표', '규격약호', '제품사내보증번호','소둔_SPM목표'])
df_origin_c = df_origin_group.count().iloc[:,0].to_frame()
df_origin_c.columns = ['count']
df_origin_c_u = df_origin_c.unstack(level='소둔_SPM목표')
df_origin_c_u.to_clipboard()

# ----------------------------------------------------------------------------------------
df_group.columns
df_group = df.groupby('출강목표')[['냉연번호생성년월','냉연코일번호']].first()
df_group.to_clipboard()

df = pd.read_clipboard()

comp = pd.read_clipboard()
comp_list = comp.columns.to_list()

df_group = df.groupby('출강목표').first()[comp_list]
df_group.to_clipboard()

