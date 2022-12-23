# [ 참고 명령어 ] ------------------------------------------------------------------------------
whos        # 메모리에 뭐가 들어갔는지 볼수 있는 명령어


# [ Function ] ------------------------------------------------------------------------------

    # Multi args Function
# def multi_args(*args):
#     print(args)           # type : tuple
#     print(type(args))
#     return sum(args)

# def f1(a, *b):     # 정상
# def f2(a, *b, c):  # 에러
# def f3(*a, *b):    # 에러
# def f4(a, b, *c):  # 정상

# def printName(first, second='Kim'):           # 정상
# def printName(first='Kim', second):           # 에러발생
# def printName(first, second, third='Kim'):    # 정상
# def printName(first, second='Kim', third='M') # 정상

# def calc(x, y, z) :
#   return x+y+z

# result = calc(y=20, x=10, z=30) # 매개변수순서상관없음
# calc(y=20, x=10) # 정상
# calc(10, y=30, z=20) # 정상
# calc(10, 30, y=20) # 에러발생



    # Multi args Fcuntion Dictionary
def abc(*arg):
    print(arg)

def abc2 (name, symbol, number):
    print(name)
    print(symbol)
    print(number)


abc(el_dict)
abc(*el_dict)
abc2(**el_dict)


# Asterisk(*, **) ===================================================
# Unpacking ---------------------------------
a = [1,2,3]
b = [4,5,6]
print(*a)
print(*b)
[*a, *b]


a = {'x':10, 'y':20}
b = {'i':30, 'j':40}
print(*a)
print(*b)



def add1(x, y):
    return x + y

def add2(i, j):
    return i + j

add1(**a)
# add2(**a)   # error: x=10, y=20

# add1(**b)   # error: i=30, j=40
add2(**b)

{**a, **b}




# Function,  ---------------------------------
def qwe(a=10, b=20, *args, **kwargs):
    print(a)
    print(b)
    print(args)
    print(kwargs)

qwe(a=10, b=20)
qwe(10, 20, 30, 40, 50, z=60)
qwe(a=10, b=20, c=30)

# qwe(a=10, 20, 30)       # error
# qwe(a=10, b=20, 30)       # error
# qwe(a=10, b=20, 30, z=60)       # error





# [ Random Module ] ------------------------------------------------------------------------------
import random
# 리스트 내용 랜덤으로 섞기
Ls = [1,2,3,4,5]
random.shuffle(Ls)      # Input List가 random하게 suffle하게끔 바뀜

# 리스트중 랜덤으로 여러개 뽑기
# random.sample(seq, N) : seq로 부터 N개의 Unique한 List를 반환함
# ① seq내의 Unique한 원소를 반환
# ② N이 Seq의 갯수를 넘어가면 Error발생 

random.sample([1, 2, 3, 4, 5, 6], 3)




# [ Time Module ] ------------------------------------------------------------------------------
import time 
time.sleep(2)       # 2초 지연


# [ Module Import ] ------------------------------------------------------------------------------
import arth     # arth.py

# if __name__ == "__main__":
#     print("직접 실행")
#     print(__name__)
# else:
#     print("임포트되어 사용됨")
#     print(__name__)

arth.add(10, 20)
arth.sub(10, 20)
arth.mul(10, 20)
arth.div(10, 20)


# [ Pythonic Comprhension ] ------------------------------------------------------------------------------
    # List Comprehension
a = [1,2,3,4,5]
b = []
for x in a:
    b.append(x*2)
b

[x*2 for x in a]

b = []
for x in a:
    if x %2 ==0:
        b.append(x)
b
[x for x in a if x % 2==0 ]


L1 = [1,2,3]
L2 = [3,4,5]

[x*y for x in L1 for y in L2]

for x in L1:
    for y in L2:
        x*y

    # enumerate
colors = ['red', 'green', 'blue']
L1 = [1,2,3]
L2 = [3,4,5]
[(i,v) for i,v in enumerate(colors) ]

    # zip
L1 = [1,2,3]
L2 = [3,4,5]
[(i,j) for i, j in zip(L1, L2)]

    # List Comprehension → Dictionary
K1 = ['One', 'Two', 'Three', 'four', 'five']
K2 = [1,2,3,4]
{k : v for k,v in zip(K1, K2)}      # 값이 적은것으로 매칭


# -----------------------------------------------------------------------------
