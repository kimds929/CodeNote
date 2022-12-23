# https://numpy.org/    # numpy 공식 문서 사이트
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# [ndarray 정보] -----------------------------------------------------------
a = np.array([[1,2],[3,4]])

print(a)
# [ [1 2]
#   [3 4] ]
print(type(a))      # <class 'numpy.ndarray'>
print(a.ndim)       # 2         # 몇차원인지?
print(a.shape)      # (2,2)     # 몇 × 몇 행렬인지?(형태)
print(a.size)       # 4         # 객체의 원소의 갯수
print(a.dtype)      # int32
print(a.itemsize)   # 4         # 객체의 원소의 사이즈??
print(a.strides)    # (8, 4)    # (1개의 row에 할당된 메모리, 1개의 원소에 할당된 메모리)
print(a.data)       # <memory at 0x000001F186ABFD68>    

    # ndarray 객체의 데이터는 itemsize 크기의 hex값으로 표현




# [차원(Dimension) : Array안에 몇개의 List 또는 Array가 있는지? ] -----------------------------------------------------------
a = np.array(10)
print(a)
print(a.ndim)      # 0 # 하나의 원소를 가지는 Array는 0차원
print(a.shape)

b = np.array([1,2,3,4])
print(b)
print(b.ndim)      # 1 # 단순 원소의 배열 List: 1차원
print(b.shape)

c = np.array([[1,2],[3,4]])
print(c)
print(c.ndim)      # 2 # List내의 List: 2차원
print(c.shape)


# reshape : np.array의 Dimension 변경 ***
    # 숫자 갯수에 맞지 않으면 에러가 발생
a = np.arange(15)
a_reshpae = a.reshape(3,5)

print(a)
print(a_reshpae)


# [행렬 생성하기] -----------------------------------------------------------
    # dataframe → numpy
df = pd.read_clipboard()
df_np = df.to_numpy() # 자동 형변환 int > float > object 으로 통일됨
    # 일반적으로 data_type을 동일한 set을 numpy로 바꾸어야 함


    # 직접 지정
x = np.array([1,3,5])
y = np.array([[2,4,6], [5,8,7]])
print(x)
print(y)
x.shape         # 3 by 1 column vector
y.shape         # 2 by 3 matrix

    # pyplot
plt.plot(y)
plt.plot(y.T)
plt.plot(x, y.T)       # pyplot


    # Transpose
print(y)
print(y.T)              # Transpose
print(np.transpose(y))  # Transpose
y.ndim
y.shape     # (row, column ...)

    # np.arange : range List 형태의 Array 생성 *** 
np.arange(10)
np.arange(1, 10)        # 1이상 10 미만
np.arange(5, 101, 5)    # 5이상 101미만까지 5단위로
np.arange(0, 20, 4)     # 0이상 20미만까지 4단위로 (20 미포함)
np.arange(10, 1, -2)    # 10이하 1초과 -1단위로


    # ones / zeros
np.ones([4,5])        # 2차원 행렬
np.ones([2, 3, 4])    # 3차원 행렬  : 2개 × (3 × 4 행렬)
np.ones((4, 5))        # Tuple로 생성
np.ones((4, 5), int)   # 원소 Type지정

a
np.ones_like(a)     # 행렬내 원소를 모두 1로 치환

np.zeros(10)
np.zeros(10, float)
np.zeros(10, int)
np.zeros((3, 2), int)

a
np.zeros_like(a)        # 행렬내 원소를 모두 zero로 치환


    # empty / full
np.empty((3, 4))     # 초기화된 값
np.full((3, 4), 7)   # 특정값으로 채우고 싶을때
np.full(10, 'a')    # 특정값으로 채우고 싶을때
np.full((2,3), 5, int)
np.full((2,3), 5, str)
np.full((2,3), 5, float)

    # np.eye : 단위행렬생성
np.eye(5)       # 5행 5열인 단위행렬
np.eye(3)       # 3행 3열인 단위행렬

    # np.linspace(start, end, n) : 동일한 차이만큼 n-1등분하여 생성 (이상, 이하)
np.linspace(1, 10, 3)     # 1이상 10이하 선형적 3분할(간격 2개) 한 값을 생성
np.linspace(0, 10, 5)     # 0이상 10이하 선형적 5분할(간격 4개) 한 값을 생성


# nonzero()
c = np.eye(5)
print(c)
c.nonzero()     # 원소의 값이 0이 아닌 원소의 위치
c[c.nonzero()]




# [random 행렬, 원소 생성] ------------------------------------------------------------------------------------
    # random Module과 비교
import random

    # [random] random.shuffle()      # List내 원소를 Shuffle
shuffle_a = np.random.randint(10, size=5)
print(shuffle_a)
random.shuffle(shuffle_a)      # Input List가 random하게 suffle하게끔 바뀜
print(shuffle_a)


    # [numpy] np.rand : 0~1사의 랜덤한 값
np.random.rand(10)      # 10개의 1차원 0~1사이의 uniform한 랜덤한 값 생성
np.random.rand(2,3)     # 2행 3열의 0~1사이의 uniform한 랜덤한 값 생성
random.random()         # (random Module) 0~1사이의 1개의 랜덤한 값 생성

    # [numpy] randn : 정규분포의 랜덤한 값 생성
np.random.randn(10)     # 10개의 1차원 평균 0, 편차 1 인 가우시안 표준정규분포 난수 생성
np.random.randn(3,4)    # 3행 4열의 평균 0, 편차 1 인 가우시안 표준정규분포 난수 생성


    # [numpy] 특정 정수 사이에서 랜덤하게 샘플링
np.random.randint(2)        # (n) : 0~n (1개)
np.random.randint(1,3)      # (n, p) : n~p (1개)
np.random.randint(1, 100, size=10)      # size=n : n개의 램덤값 생성
np.random.randint(1, 100, size=(3,5))   # size=(n, p) : (n, p)개의 램덤값 생성
random.randint(1,10)        # (random Module) (a, b) : a~b까지의 값중에 1개의 랜덤한 값 생성


    # [numpy] choice : 주어진 1차원 ndarray로부터 랜덤으로 샘플링, 정수가 주어진 경우 np.arange(숫자)로 간주
np.random.choice(100, size=(3,4))               # 0이상 100미만의 값들중에 size 만큼의 array생성

x = np.array([1,2,3,1.5,2.6, 4.9])
np.random.choice(x, size=(2,3))                 # x의 List값중에 size만큼의 원소를 뽑아서 array생성
np.random.choice(x, size=(2,3), replace=False)  # replace : 중복허용여부

    # [random] random.sample 리스트중 랜덤으로 여러개 뽑기
        # random.sample(seq, N) : seq로 부터 N개의 Unique한 List를 반환함
        # ① seq내의 Unique한 원소를 반환
        # ② N이 Seq의 갯수를 넘어가면 Error발생 
sample_a = np.random.randint(10, size=5)
print(sample_a)
# random.sample(sample_a, 3)      # Error
print( random.sample(list(sample_a), 3) )


    # [numpy] permutation
np.random.permutation(10)   # 0~9까지 문자를 random하게 섞어주는 shuffle을 만들어 주는것

# permutation shuffle example
a = np.arange(0,10)
b = np.arange(10,20)
idx = np.random.permutation(10)   # 0~9까지 문자를 random하게 섞어주는 shuffle을 만들어 주는것
a[idx]
b[idx]



    # [numpy] 확률분포에 따른 ndarray 생성 : uniform, normal
a = np.random.uniform(low=1, high=100, size =50)      # 1~100까지 균일하게 50개
b = np.random.normal(loc=50, scale=10, size=50)       # 평균: 50, 편차: 10, 갯수: 50개
plt.hist(a)
plt.hist(b)

np.random.uniform(low=1, high=2, size=(2,4))
np.random.normal(loc=100, scale=10, size=(2,4))        # 평균 : 100, 편차 10


    # [numpy] seed : 랜덤한 값을 동일하게 다시 생성하고자 할때 생성
np.random.seed(100)
print(np.random.randn(3,4))
print(np.random.randn(3,4))
print(np.random.randn(3,4))




# [인덱싱 및 슬라이싱] ------------------------------------------------------------------------------------
    # 1차원 행렬 인덱싱
x = np.arange(10)
x
x[0]
x[len(x)-1]     # 마지막값 추출
x[-1]           # 마지막값 추출

x[3] = 100  # 값변경
x

    # 2차원 행렬 인덱싱
y = np.arange(10).reshape(2,5)
y
y[0]        # 2개의 차원중 첫번째 차원 array
y[1][4]     # 2번째 차원내의 4번 index값(5번째)
y[1,4]      # 2번째 차원내의 4번 index값(5번째)

    # 3차원 행렬 인덱싱
z = np.arange(36).reshape(3,4,3)
z
z[0]
z[0][0]
z[0][0][0]
z[0,0,0]

    # 슬라이싱
x = np.arange(10)
x
x[1:7]
x[[1,2]]

y = np.arange(10).reshape(2,5)
y
y[1][:2]
y[1,[1,3]]
y[1,1:4]
y[1,:4]

z = np.arange(36).reshape(3,4,3)
z
z[0,1,1]
z[0,[2,3],1]
z[0,[2,3],:2]



    # Array indexing  ***
a = np.array([[10,20,30], [40.5, 50, 60]], int)
a           
# [ [10, 20, 30], 
#   [40, 50, 60] ]

a[0]        # [10 20 30]
a[0][0]     # 10

a[0,0]      # 10
a[0,:]      # [10, 20, 30]
a[:,0]      # [10, 40]


# Numpy Array 타입의 검색이나 슬라이싱은 '참조만 할당' 하르모 변경을 방지하기 위해서는
# 새로운 ndarray로 만들어서 사용 → .copy() 메소드가 필요

l = [1, 2, 3, 4]
la = np.array(l)

print(la)
s = la[:3]
ss = la[:3].copy()

s[0] = 99       # 참조된 s변수의 0번 index 값을 변경
ss[1] = 100      # 복사된 ss변구의 1번 index 값을 변경

print(la[:3])   # 참조된 s변수의 값 변경에 따라 원본도 바뀜, 복사된 ss값의 변경은 원본에 영향을 미치지 않음
print(s)        # 참조된 s변수 결과
print(ss)       # 복사된 ss변수 결과



    # matrix 대각 행렬값만 추출
mat_a = np.arange(9).reshape(3,3)
print(mat_a)
np.diag(mat_a, k=0)  # k - start index
np.diag(mat_a, k=1)  # k - start index
np.diag(mat_a, k=2)  # k - start index

np.diag(mat_a, k=-1)  # k - start index
np.diag(mat_a, k=-2)  # k - start index



    # 다중인덱싱
a = np.arange(1,10).reshape(3,3)
print(a)
print(a[[0,1,2,0], [0,2,0,1]])      # [[row좌표], [column좌표]]


y = np.arange(35).reshape(5,7)
print(y)
print( y[1:5:2, ::3] )      
    # row: 1~5까지 2단위로 끊어서 indexing
    # column: 0~끝까지 3단위로 끊어서 indexing



    # List Array Transformation ***
a = range(1, 10)                # List 
np.array(a)                     # List → Array
np.array(a, float)              # List → Array(float)
np.array(a, dtype=np.float_)    # List → Array(float)

    # __getitem__ : Array Indexing   ***
a_array = np.array(a)
a_array.dtype

a.__getitem__(0)
a_array.__getitem__([0,5])
# a_array[[0,5]]

b = np.array([x>3 for x in a])
a_array.__getitem__(b)
# a_array[b]
# a_array[ a_array >3 ]





    # __setitem__ : 원소 치환   ***
# b_array = a_array         # shallow copy
# b_array = a_array[:]      # shallow copy
b_array = a_array.copy()    # deep copy

b_array.__setitem__(1,100)
# b_array[1] = 100
b_array
a_array

b_array = a_array.copy()    # deep copy
b_array.__setitem__(b, 100)
# b_array[b] = 100
b_array
a_array





# [ndarray 데이터 형태 바꾸기(reshape, flatten)] -----------------------------------------------------------
    # ravel : 다차원 → 1차원
z = np.arange(36).reshape(3,4,3)
print(z)
z.ravel()
np.ravel(z)

z_r = z.copy()
print(z_r)
z_ravel = z_r.ravel()
z_ravel
z_ravel[0] = 100

z_ravel
z_r     # z_ravel 의 0번째 값을 바꿨는데 z_r 0번째 값도 같이 바뀌었음 (List = Mutable)

z.ravel(order='c')  # order='c' : row 우선변경
z.ravel(order='f')  # order='f' : column 우선변경

    # flatten : 다차원 → 1차원
z.flatten()
# np.flatten(z)

z_f = z.copy()
z_flatten = z_f.flatten()
z_flatten
z_flatten[0] = 100
z_flatten
z_f     # z_flatten 의 0번째 값이 바뀌어도 z_f행렬에 영향을 미치지 않음 (copy를 생성하여 변경함)


z.flatten(order='c')  # order='c' : row 우선변경
z.flatten(order='f')  # order='f' : column 우선변경


    # reshape : 1차원 → 다차원
x = np.arange(36)
x.ndim     # 차원 = 1
len(x)
y = x.reshape(6,6)
x.reshape(6,-1)     # -1 : 나머지 값을 알아서 정해줌
x.reshape(-1, 9)     # -1 : 나머지 값을 알아서 정해줌

y.shape     # 형태
y.ndim      # 차원




# [array, matrix Operation] ------------------------------------------------------------------------------------
x = np.arange(15).reshape(3,5)
y = np.random.randint(1, 30, size=15).reshape(3,5)
z = y.reshape(5,3) 
print(x)
print(y)
print(z)

    # 기본연산함수
np.add(x,y)         # add
x + y

np.subtract(x,y)    # subtract
x - y

np.multiply(x,y)    # multiply
x * y

np.divide(x,y)      # divide
x / y

np.add(x,z) # 행렬의 shape이 같지 않은경우에는 연산 불가
x.shape == z.shape
x.shape == y.shape



    # 연산자 기능
height = [1.73, 1.68, 1.71, 1.89, 1.79]
np_height = np.array(height)

weight = [65.4, 59.2, 63.6, 88.4, 68.7]
np_weight = np.array(weight)

    # Plus(+) 연산자 ***
height + weight         # List 끼리의 합은 하나의 리스트에 원소가 추가되는 효과만 있음
np_height + np_weight   # Array 끼리의 합은 대응하는 위치의 원소끼리 합을 구해줌

    # Type 변경 ***
np.array(height, int)       # List 각 원소의 Type을 지정하여 Array로 반환
np.array(np_height, int)      # Array 각 원소의 Type을 지정하여 Array로 반환

    # 연산자 비교 (List vs Array vs Matrix)
a0 = [[1,2],[3,4]]
b0 = [[2,2],[2,2]]

a1 = np.array([[1,2],[3,4]])
b1 = np.full((2,2),2, int)

a2 = np.matrix(a1)
b2 = np.matrix(b1)

print('-- array --')
print(a1)
print(b1)
print('-- matrix --')
print(a2)
print(b2)

print('-- array multiply--')
print(a1 * b1)               # 원소간 곱
print(np.multiply(a1, b1))   # 원소간 곱
print(a1 @ b1)               # 행렬곱
print(np.dot(a1, b1))        # 행렬곱
# print(a1.dot(b1))


print('-- matrix multiply--')
print(a2 * b2)               # 행렬곱
print(np.multiply(a2, b2))   # 원소간 곱
print(a2 @ b2)               # 행렬곱
print(np.dot(a2, b2))        # 행렬곱
# print(a2.dot(b2))




# Operator (+, -, /)
a0 + b0     # 리스트 원소 추가 효과
a1 + b1
a2 + b2

a0 - b0     # Error
a1 - b1
a2 - b2

a0 / b0     # Error
a1 / b1
a2 / b2


# [결측치 처리] ------------------------------------------------------------------------------------
df = pd.read_clipboard()
df_np = df.to_numpy() 

np.isnan(df_np)  # 결측치 확인

idx_missing = np.isnan(df_np)
print(idx_missing)

print('전체 결측치의 개수:', np.isnan(df_np).sum()) # 전체 결측치 갯수 확인
print('열별 결측치의 개수:', np.isnan(df_np).sum(0)) # column(열)별 결측치 갯수 확인
print('행별 결측치의 개수:', np.isnan(df_np).sum(1)) # row(행)별 결측치 갯수 확인

    
    # 실습 : 결측치가 200,000개 이상인 열들의 전체 평균과 결측치가 200,000개 미만인 열들의 전체 평균 비교하기
idx_col1 =np.isnan(df_np).sum(0) >=200000
df_contin_np_20up = df_np[:,idx_col1]

idx_col2 =np.isnan(df_np).sum(0) <200000
df_contin_np_20down = df_np[:,idx_col2]


print(f'결측칙 20만개 미만 평균 : {np.nanmean(df_contin_np_20down)}')
print(f'결측칙 20만개 이상 평균 : {np.nanmean(df_contin_np_20up)}')


    # 결측치가 50% 이상인 열 제외
Num_obs = df_np.shape[0]
Num_missing_by_column = idx_missing.sum(0)
idx_col_valid = Num_missing_by_column<(Num_obs * 0.5) # 열별 결측치의 수 < 전체 관측치의 수 
print(idx_col_valid)

df_np2 = df_np[:,idx_col_valid]
print(df_np2)


# [기본 내장함수 및 통계함수] ------------------------------------------------------------------------------------
    # Numpy 수치 함수
np.pi
np.exp(1)

np.tan(1)
np.sin(0)
np.cos(0)

np.tanh()
np.sinh()
np.cosh(0)



    # 통계 함수
x = np.arange(15).reshape(3,5)
y = np.random.randint(1, 30, size=15).reshape(3,5)
z = y.reshape(5,3) 
print(x)
print(y)
print(z)

y.mean()
np.mean(y)

y.std()
np.std(y)
y.var()
np.var(y)

y.max()
np.max(y)
np.argmax(y)    # max값이 존재하는 위치의 index (flatten한 상태로 가정)

y.min()
np.min(y)
np.argmin(y)    # min값이 존재하는 위치의 index (flatten한 상태로 가정)

np.median(y)    # y.median() Error


    # 집계 함수
y
y.sum()     # 합계
np.sum(y)
np.sum(y, axis=0)
np.sum(y, axis=1)
z
np.sum(z, axis=0)
np.sum(z, axis=1)
np.sum(z, axis=2)
np.sum(z, axis=-1)
np.sum(z, axis=(1,2))

y.cumsum()      # 누적합
np.cumsum(y)

y.prod()    # 전체곱
np.prod(y)


    # any : 특정조건을 하나라도 만족하는 것이 있으면 True, 아니면 False
x
np.any(x > 10)
np.any(x < 0)

    # all : 특정조건을 모두 만족하면 True, 아니면 False
np.all(x > 10)
np.all(x > -1)


    # where : 조건에 따라 선별적으로 값을 선택
y = np.array([[ 6, 29, 28, 15,  2], [15, 26, 10,  2, 27], [14, 18,  7, 28, 16]])
y[y>10] = 999

y = np.array([[ 6, 29, 28, 15,  2], [15, 26, 10,  2, 27], [14, 18,  7, 28, 16]])
np.where(y >10, 999, 0)     # (조건, True, False)
np.where((y >10) & (y < 30), 999, 0)     # (조건, True, False)


# ?np.argwhere  # True에 해당하는 값의 좌표값(r,c)을 반환 --------------
# get_ipython().run_line_magic('pinfo', 'np.argwhere')

# a = np.array([False, False, True])
# np.argwhere(a)      # 2

# a = np.array([[False, False, True],[True,True,False]])
# np.argwhere(a)  
# array([[0, 2], [1, 0], [1, 1]], dtype=int64)      # True의 위치 (0,2), (1,0), (1,1)
# ---------------------------------------------------------------------




# broadcasting : shape이 다른경우 shape을 맞춰주는 방법
# 뒷 차원에서부터 비교하여 shape이 같거나, 차원 중 값이 1인 것이 존재하면 가능

    # matrix broadcasting possible? 
#  * 뒤를 기준으로 붙여서 규칙이 맞으면 가능
#     # 하나가 1이거나, 숫자가 같으면 가능
    # https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

bx = np.arange(15).reshape(3,5)
by = np.random.randint(10,30,size=15).reshape(3,5)
bx
by

    # broadcasting: shape이 같은경우
bx + by

    # broadcasting: 상수항과의 연산
bx + 2
bx * 2
bx ** 2
bx % 2

    # broadcasting: shape이 다른경우 연산
rx = np.arange(12).reshape(4,3)
rx2 = np.arange(12).reshape(3,4)
rx
rx2

ry = np.arange(20,23)
ry2 = ry.reshape(1,3)
ry
ry2.shape

rz = np.arange(30,34)
rz

rx + ry
rx + rz  # error
rx2 + rz
rx + ry2


# 차원 확장 & 축소 ----------------------------------------------------------
np_dim = np.random.rand(3,1,2)
print(np_dim)
print(np_dim.shape)


# 차원추가
print(np.expand_dims(np_dim, 0).shape)
print(np.expand_dims(np_dim, 1).shape)
print(np.expand_dims(np_dim, 2).shape)
print(np.expand_dims(np_dim, 3).shape)
print(np.expand_dims(np_dim, -1).shape)

print(np_dim[np.newaxis, ...].shape)
print(np_dim[..., np.newaxis].shape)


# 차원축소
print(np.squeeze(np_dim).shape)

np_dim2 = np.random.rand(3,1,2,1,2)
print(np_dim2)
print(np_dim2.shape)
print(np.squeeze(np_dim2).shape)


# 1차원화
print(np_dim.reshape(-1))
print(np_dim.reshape(-1).shape)

print(np_dim.ravel())
print(np_dim.ravel().shape)

print(np_dim.flatten())
print(np_dim.flatten().shape)






# [boolean Indexing] ------------------------------------------------------------------------------------
x = np.random.randint(1,100, size=10)
x

x_mask = x % 2 == 0
x_mask
x[x_mask]
x[x % 2 == 0]
x[x > 30]

    # 다중조건 사용하기: and → &,  or → |
x[(x % 2 ==0) & (x < 30)]
x[(x % 2 ==0) | (x < 30)]



# [linalg : 선형대수 연산] ------------------------------------------------------------------------------------
    # np.linalg.inv : 역행렬을 구할때 사용(모든 차원의 값이 같아야 함)
x = np.random.randint(10, 20, size=(3,3))
x

x_inv = np.linalg.inv(x)
x_inv

x @ x_inv               # 행렬곱 (= matmul)
    # matmul(matrix multiplication) : Matrix product of two arrays. (행렬을 곱하는 것을 행렬곱)
np.matmul(x, x_inv)     # 행렬곱 (2차원까지는 dot와 동일)

    # dot(tensor product) : Dot product of two arrays. (두 텐서를 곱하는 것을 텐서곱)
np.dot(x, x_inv)        # 행렬곱 (2차원까지는 matmul과 동일)

    # np.linalg.solve : Ax=B형태의 선형대수식 솔루션을 제공 ( B · A^-1 )
        # x + y = 25, 2x + 4y = 64
        # [[1, 1], [2, 4]] × [x, y] =  [25, 64]
a = np.array([[1,1], [2,4]])
b = np.array([25,64])
x = np.linalg.solve(a,b)
x

    # allclose
np.allclose(a @ x, b)       # 거의같은경우 값을 비교하기 위한 함수



    # eigen value, vector 
x
np.linalg.eig(x)[0] # eigen value
np.linalg.eig(x)[1] # eigen vector








# Numpy Array Delete Row, Column ----------------------
a0 = np.array([[1, 2], [3, 4]])

np.delete(arr=a0, obj=0, axis=1)    # column 제거
np.delete(arr=a0, obj=0, axis=0)    # row 제거


# Numpy Array Concatenation ----------------------
    # numpy array concatenate
a0 = np.array([[1, 2], [3, 4]])
b0 = np.array([[5, 6]])

np.concatenate((a0, b0), axis=0)        # (default) axis=0
# array([[1, 2],
#        [3, 4],
#        [5, 6]])

np.concatenate((a0, b0.T), axis=1)
# array([[1, 2, 5],
#        [3, 4, 6]])

np.concatenate((a0, b0), axis=None)
# array([1, 2, 3, 4, 5, 6])


    # vstack, hstack, dstack
a1 = np.random.randint(50, 80, size=10)
b1 = np.random.randint(1, 30, size=10)

np.concatenate((a1, b1), axis=0)
# np.concatenate((a.T, b.T), axis=1)      # error :  axis 1 is out of bounds for array of dimension 1

np.hstack((a1, b1))     # axis=0
np.vstack((a1, b1))     # axis=1
np.vstack((a1, b1)).T

np.dstack((a1, b1))


    # additional...
c = np.concatenate((a0, b0), axis=0) 
c0 = c[:,0]
c1 = c[:,1]
c_list = [c0, c1]
print(c)
print(c0)
print(c1)
print(c_list)

np.vstack(c_list).T
np.dstack(c_list)




# 정렬 (Sort)

# - (1) 1차원 배열 정렬 : np.sort(x)
# - (2) 1차원 배열 거꾸로 정렬 : np.sort(x)[::-1] , x[np.argsort(-x)]
# - (3) 2차원 배열 열 축 기준으로 정렬 : np.sort(x, axis=1)
# - (4) 2차원 배열 행 축 기준으로 정렬 : np.sort(x, axis=0)
# - (5) 2차원 배열 행 축 기준으로 거꾸로 정렬 : np.sort(x, axis=0)[::-1]






# EINSUM (Einstein summation) ***
# matrix (i × j) → summation axis
import numpy as np

# An Matrix ***
a1 = np.array( [[1,2,3],
               [4,5,6]])
a2 = np.array( [[-1,-2],
               [-3,-4],
               [-5,-6]] )

np.einsum('ij->ji', a1)  # transpose
np.einsum('ij->',a1)     # summation
np.einsum("ij->i", a1)   # row sum  (= a1.sum(1) )
np.einsum("ij->j", a1)   # column sum (= a1.sum(0) )


e1 = np.array( [[1,2,3],
                [4,5,6],
                [7,8,9]] )
np.einsum('ii->i', e1)   # diag
np.einsum('ii->', e1)    # trace (diagonal summation)


# Multiplication ***  Dot Product, Outer product of two vectors
b1 = np.array([1,2,3])
b1T = b1.reshape(-1,1)

b2 = np.array([-1,-2,-3])
b2T = b2.reshape(-1,1)

np.einsum('i,i->', b1, b2 )     # dot (= b1@b2 = (b1*b2).sum())
np.einsum('i,j->ij', b1,b2)     # outer (= b1 * b2.reshape(-1,1) )

np.einsum('ij,jk->ik',a1,a2)        # dot product
# np.einsum('ij,jk->',a1,a2)        # dot product

# Hadamard(element-wise) product of vector or matrix

np.einsum('i,i->i', b1, b2)     # elementwise multiply (= b1 * b2 )
np.einsum('ij,ij->',a1,a1)      # elementwise multiply (= (a1*a1).sum() )


np.einsum('ij,j->i', a1, b1)
np.einsum('ij,j->j', a1, b1)

np.einsum('ij,i->i', a2, b1)
np.einsum('ij,i->j', a2, b1)

np.einsum('i,ji->', b1,a1)
np.einsum('i,ji->i', b1,a1)
np.einsum('i,ji->j', b1,a1)

np.einsum('i,ij->', b1,a2)
np.einsum('i,ij->i', b1,a2)
np.einsum('i,ij->j', b1,a2)


ab_ijk =np.einsum('i,jk->ijk', b1,a2)     # 

np.einsum('i,jk->jk', b1,a2)     # ab_ijk.sum(axis=0)
np.einsum('i,jk->ik', b1,a2)     # ab_ijk.sum(axis=1)
np.einsum('i,jk->ij', b1,a2)     # ab_ijk.sum(axis=2)

np.einsum('i,jk->k', b1,a2)     # ab_ijk.sum(axis=(0,1))
np.einsum('i,jk->j', b1,a2)     # ab_ijk.sum(axis=(0,2))
np.einsum('i,jk->i', b1,a2)     # ab_ijk.sum(axis=(1,2))

np.einsum('i,jk->', b1,a2)     # ab_ijk.sum(axis=(0,1,2))


a3 = np.stack([a1,a1*2,a1*3, a1*4]).transpose(1,2,0)
a3.shape

np.einsum('ij,jkl->ijkl', a2, a3)
np.einsum('ij,jkl->ijkl', a2, a3).shape
np.einsum('ij,jkl->ijkl', a2, a3).sum(axis=(1,2))
np.einsum('ij,jkl->il', a2, a3)

a2[0][0] * a3[0]
a2[0][1] * a3[1]







# [연습문제1 : 로또번호 추출] ------------------------------------------------------------------------------------

def generate_lotto_nums():
    return np.random.choice(np.arange(1,46), size=6, replace=False)

generate_lotto_nums()



# [연습문제2 : ] ------------------------------------------------------------------------------------


