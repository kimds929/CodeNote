import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
 
from math import pi
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
 
## 데이터 준비
df = pd.DataFrame({
'Character': ['Barbarian','Amazon','Necromancer','Sorceress','Paladin'],
'Strength': [10, 5, 3, 2, 7],
'Dexterity': [4, 10, 3, 3, 8],
'Vitality': [9, 9, 7, 7, 8],
'Energy': [4, 4, 10, 10, 6],
'Wisdom': [2, 6, 8, 9, 8],
'Total': [4, 7, 3, 9, 5]
})



df_origin = pd.read_clipboard(sep='\t')
group = '강종'

# angles[:-1]


df_origin_rev = df_origin.copy()

for c in ['실패비', '가공비', '원료비', '제조원가']:
    df_origin_rev[c] = -df_origin_rev[c]

criteria = {
'판매량' : [0, 50]
, '영업이익률' : [-20, 20]
, '판매가' : [1200, 1800]
, '제조원가' : [-1800, -1200]
, '원료비' : [-1100, -800]
, '가공비' : [-900, -400]
, '실패비' : [-200, 0]
, '실수율' : [60, 90]
}

criteria_points = {k: np.linspace(v[0],v[1]+1e-10, 6)[1:] for k, v in criteria.items()}

np.linspace(-30, 30, 6)


df = df_origin_rev.copy()
for c in df_origin_rev:
    if c in criteria.keys():
        df[c] = (df[c] - criteria[c][0]) / (criteria[c][1] - criteria[c][0]) *10
        df[c] = df[c].apply(lambda x: 0 if x < 0 else (10 if x > 10 else x))


            



# from sklearn.preprocessing import MinMaxScaler
# mm = MinMaxScaler()
# mm.fit(df_origin_rev.drop(group, axis=1))
# df_transform = mm.transform(df_origin_rev.drop(group, axis=1))
# df = pd.DataFrame(df_transform, columns=df_origin.drop(group, axis=1).columns) *5+5
# df.insert(0, group, df_origin[group])



## 따로 그리기 ----------------------------------------------------------------
labels = df.columns[1:]
num_labels = len(labels)
    
angles = [x/float(num_labels)*(2*pi) for x in range(num_labels)] ## 각 등분점
angles += angles[:1] ## 시작점으로 다시 돌아와야하므로 시작점 추가

# my_palette = plt.cm.get_cmap("Set2", len(df.index))
my_palette = ['navy','royalblue','skyblue', 'brown', 'gold',
              'mediumseagreen', 'navy', 'royalblue', 'limegreen', 'mediumseagreen',
              'royalblue', 'skyblue' , 'coral', 'brown', 'mediumseagreen',
            'yellowgreen']

# ('CR 980DP 高YS', 'CR 980DP 低Ceq', 'CR 980DP 고연신', 'CR 1180 TRIP', 'CR 980XF', 
#  'CR 1180CP 高YS', 'GI 980DP 高YS', 'GI 980DP 低Ceq', 'GI 980CP', 'GI 1180CP 高YS', 
#  'GA 980DP 低Ceq', 'GA 980DP 고연신', 'GA 980 TRIP', 'GA 1180 TRIP', 'GA 1180CP 高YS', 
#  'GA 1180CP 高EL')

 
fig = plt.figure(figsize=(20,40))
fig.set_facecolor('white')
 
for i, row in df.iterrows():
    # color = my_palette(i)
    color = my_palette[i]
    data = df.iloc[i].drop(group).tolist()
    data += data[:1]
    
    ax = plt.subplot(7,3,i+1, polar=True)
    ax.set_theta_offset(pi / 2) ## 시작점
    ax.set_theta_direction(-1) ## 그려지는 방향 시계방향
    
    plt.xticks(angles[:-1], labels, fontsize=13, weight='bold') ## x축 눈금 라벨
    ax.tick_params(axis='x', which='major', pad=15) ## x축과 눈금 사이에 여백을 준다.
 
    ax.set_rlabel_position(0) ## y축 각도 설정(degree 단위)
    # plt.yticks([0,2,4,6,8,10],['0','2','4','6','8','10'], fontsize=10) ## y축 눈금 설정
    plt.yticks(alpha=0)
    plt.ylim(0,10)
    
    ax.plot(angles, data, color=color, linewidth=2, linestyle='solid') ## 레이더 차트 출력
    ax.fill(angles, data, color=color, alpha=0.2) ## 도형 안쪽에 색을 채워준다.
    
    plt.title(row[group], size=20, color=color, x=-0.2, y=1.1, ha='left') ## 타이틀은 캐릭터 클래스로 한다.

    # index
    for a_idx, angle in enumerate(angles[:-1]):
        for p_idx, value in enumerate(criteria_points[list(criteria.keys())[a_idx]]):
            plt.text(angle, p_idx*2+2, auto_formating(value, decimal_revision=-1, return_type='str'), fontsize=10, alpha=0.2)
    
    for angle, data, value in zip(angles[:-1], data[:-1], list(df_origin.iloc[i].drop(group))):
        plt.text(angle, data, auto_formating(value, decimal_revision=-1, return_type='str'), fontsize=15, weight='bold')  
plt.tight_layout(pad=3) ## subplot간 패딩 조절
plt.show()




## 따로 그리기 ----------------------------------------------------------------
labels = df.columns[1:]
num_labels = len(labels)
    
angles = [x/float(num_labels)*(2*pi) for x in range(num_labels)] ## 각 등분점
angles += angles[:1] ## 시작점으로 다시 돌아와야하므로 시작점 추가

# my_palette = plt.cm.get_cmap("Set2", len(df.index))
my_palette = ['navy','royalblue','skyblue', 'brown', 'gold',
              'mediumseagreen', 'navy', 'royalblue', 'limegreen', 'mediumseagreen',
              'royalblue', 'skyblue' , 'coral', 'brown', 'mediumseagreen',
            'yellowgreen']

# ('CR 980DP 高YS', 'CR 980DP 低Ceq', 'CR 980DP 고연신', 'CR 1180 TRIP', 'CR 980XF', 
#  'CR 1180CP 高YS', 'GI 980DP 高YS', 'GI 980DP 低Ceq', 'GI 980CP', 'GI 1180CP 高YS', 
#  'GA 980DP 低Ceq', 'GA 980DP 고연신', 'GA 980 TRIP', 'GA 1180 TRIP', 'GA 1180CP 高YS', 
#  'GA 1180CP 高EL')

 
plots = []
for i, row in df.iterrows():
    # color = my_palette(i)
    fig = plt.figure(figsize=(7,7))
    fig.set_facecolor('white')
    ax = plt.subplot(polar=True)
    
    color = my_palette[i]
    data = df.iloc[i].drop(group).tolist()
    data += data[:1]
    
    # ax = plt.subplot(7,3,i+1, polar=True)
    ax.set_theta_offset(pi / 2) ## 시작점
    ax.set_theta_direction(-1) ## 그려지는 방향 시계방향
    
    plt.xticks(angles[:-1], labels, fontsize=13, weight='bold') ## x축 눈금 라벨
    ax.tick_params(axis='x', which='major', pad=15) ## x축과 눈금 사이에 여백을 준다.
 
    ax.set_rlabel_position(0) ## y축 각도 설정(degree 단위)
    # plt.yticks([0,2,4,6,8,10],['0','2','4','6','8','10'], fontsize=10) ## y축 눈금 설정
    plt.yticks(alpha=0)
    plt.ylim(0,10)
    
    ax.plot(angles, data, color=color, linewidth=2, linestyle='solid') ## 레이더 차트 출력
    ax.fill(angles, data, color=color, alpha=0.2) ## 도형 안쪽에 색을 채워준다.
    
    plt.title(row[group], size=20, color=color, x=-0.2, y=1.1, ha='left') ## 타이틀은 캐릭터 클래스로 한다.

    # index
    for a_idx, angle in enumerate(angles[:-1]):
        for p_idx, value in enumerate(criteria_points[list(criteria.keys())[a_idx]]):
            plt.text(angle, p_idx*2+2, auto_formating(value, decimal_revision=-1, return_type='str'), fontsize=10, alpha=0.2)
    
    for angle, data, value in zip(angles[:-1], data[:-1], list(df_origin.iloc[i].drop(group))):
        plt.text(angle, data, auto_formating(value, decimal_revision=-1, return_type='str'), fontsize=15, weight='bold')  
    plt.tight_layout(pad=3) ## subplot간 패딩 조절
    plt.close()
    
    plots.append(fig)


plots_iter = iter(plots)
img_to_clipboard(next(plots_iter), dpi=100)




## 하나로 합치기 ----------------------------------------------------------------
labels = df.columns[1:]
num_labels = len(labels)
    
angles = [x/float(num_labels)*(2*pi) for x in range(num_labels)] ## 각 등분점
angles += angles[:1] ## 시작점으로 다시 돌아와야하므로 시작점 추가
    
# my_palette = plt.cm.get_cmap("Set2", len(df.index))
# my_palette = ['steelblue','mediumseagreen','darkviolet','coral', 'gold']
my_palette = ['navy','royalblue','skyblue', 'brown', 'gold',
              'mediumseagreen', 'navy', 'royalblue', 'limegreen', 'mediumseagreen',
              'royalblue', 'skyblue' , 'coral', 'brown', 'mediumseagreen',
            'yellowgreen']

 
fig = plt.figure(figsize=(8,8))
fig.set_facecolor('white')
ax = fig.add_subplot(polar=True)
for i, row in df.iterrows():
    # color = my_palette(i)
    color = my_palette[i]
    data = df.iloc[i].drop(group).tolist()
    data += data[:1]
    
    ax.set_theta_offset(pi / 2) ## 시작점
    ax.set_theta_direction(-1) ## 그려지는 방향 시계방향
    
    plt.xticks(angles[:-1], labels, fontsize=13) ## 각도 축 눈금 라벨
    ax.tick_params(axis='x', which='major', pad=15) ## 각 축과 눈금 사이에 여백을 준다.
 
    ax.set_rlabel_position(0) ## 반지름 축 눈금 라벨 각도 설정(degree 단위)
    # plt.yticks([0,2,4,6,8,10],['0','2','4','6','8','10'], fontsize=10) ## 반지름 축 눈금 설정
    plt.yticks(alpha=0)
    plt.ylim(0,10)
    ax.plot(angles, data, color=color, linewidth=2, linestyle='solid', label=row[group]) ## 레이더 차트 출력
    # ax.fill(angles, data, color=color, alpha=0.3) ## 도형 안쪽에 색을 채워준다.
    plt.grid(alpha=0.3)


# index
for a_idx, angle in enumerate(angles[:-1]):
    for p_idx, value in enumerate(criteria_points[list(criteria.keys())[a_idx]]):
        plt.text(angle, p_idx*2+2, auto_formating(value, decimal_revision=-1, return_type='str'), fontsize=13, alpha=0.3)

for angle, data, value in zip(angles[:-1], data[:-1], list(df_origin.iloc[i].drop(group))):
    plt.text(angle, data, auto_formating(value, decimal_revision=-1, return_type='str'), fontsize=15, weight='bold')  
    
plt.legend(loc=(1,1))
# plt.legend(bbox_to_anchor=(1,1))
plt.show()



