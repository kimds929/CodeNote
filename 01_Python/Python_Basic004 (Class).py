# [Class 기초]-----------------------------------------------------
# Class 생성자 실습
class Complex:
    def __init__(self, realpart, imagpart):
        self.r = realpart
        self.i = imagpart

x = Complex(realpart=10, imagpart=20)
x.r, x.i

# Slime 만들기
class Slime:
    def __init__(self, name='slime', hp=100):
        self.name=name
        self.hp=hp

    def acttack(self, damage):
        self.hp = self.hp - damage


a = Slime(name='Slime_A', hp=500)
a.name
a.hp

a.acttack(300)

b = Slime()
b.name
b.hp

# 인스턴스 객체
    # 인스턴스 객체는 오직 한가지 연산만 가능
    # 데이터 어트리뷰트는 처음 대입될 때 생성, 선언 필요

# 메서드 객체
    # 일반 함수 객체와 달리, 클래스에 종속되어 있는 함수를 지칭
    # 대상이 되는 객체가 있어야 사용 가능

# 클래스 = 틀
# 인스턴스 = 객체

# 인스턴스 데이터 = 인스턴스가 가지는 데이터
# 클래스 데이터 = 클래스의 모든 인스턴스가 공유하는 어트리뷰트와 메서드를 지칭
    # 아래와 같이 kind의 값의 수정은 불가능 (immutable 불가능)
    # 변경시 해당 인스턴스에 새로운 변수가 복사, 생성됨
    # 단 mutable 객체의 경우 직접 수정 가능 (주의)
class Dog:
    kind = 'canine'         # 클래스 데이터 
    # immutable : 수정은 불가하나, 변경시 새로운 변수가 복사/생성됨 (수정된 것처럼 보임)
    # muttable : 직접 수정 가능

    def __init__(self, name):
        self.name = name        # 인스턴스 데이터

a = Dog('abc')
a.name = 'bcd'
a.kind = 'aaa'
a.kind



# [클래스 상속] -------------------------------------------------------------------------------
    # 기존 클래스의 속성을 물려 받아 새로운 클래스를만드는 것
    # 새 클래스는 기존 클래스의 모든 변수 및 메서드를 가짐
    # 주로 기존 클래스를 확장하는 용도로 사용
    # 다중 클래스도 상속 가능 ”,”로 구분
# class Derived_Class(Super_Class):
#     pass

# https://dojang.io/mod/page/view.php?id=2386


# Slime 만들기 -----------------
class Slime:
    def __init__(self, name='slime', hp=100):
        self.name=name
        self.hp=hp

    def acttack(self, damage):
        self.hp = self.hp - damage

# Slime 상속
class Slime_magic(Slime):
    def __init__(self, mp=50):
        super().__init__()      # super()로 기반 클래스의 __init__ 메서드 호출
        self.mp = mp

    def fireball(self):
        self.mp = self.mp - 10
        return 5


# Slime 상속        # 파생 클래스에 __init__ 메서드가 없다면 기반 클래스의 __init__이 자동으로 호출
class Slime2(Slime):
    pass


a = Slime_magic()
a.name
a.hp

a.fireball()
a.mp

b = Slime_magic() 
b.hp
b.acttack(a.fireball())
b.hp

a.mp

c = Slime2()
c.hp




# 국가  -----------------
class Country:      # Super Class : 상속하는 클래스
    name = '국가명'
    population = '인구'
    capital = '수도'

    def show(self):
        print('국가 클래스의 메소드입니다.')


class Korea(Country):   # Sub Class : 상속받는 클래스
    def __init__(self, name):
        self.name = name

    def show_name(self):
        print('국가 이름은 : ', self.name)

a = Korea('한국')
a.show_name()       # Sub Class에서 정의된 메서드
a.name              # Super Class에서도 정의되었으나 Sub Class에서 다시 정의하여 변경된 내용 

a.population        # Sub Class에는 없는 속성이나, Super Class를 상속함으로서 Super Class를 사용할 수 있음
a.capital           # Sub Class에는 없는 속성이나, Super Class를 상속함으로서 Super Class를 사용할 수 있음



# [메서드 오버라이딩 (Overriding)] ------------------------------------------------------------
    # 기존 클래스의 메서드를 새로운 클래스에서 재정의 하는 것
    # 재정의하지 않은 메서드는 기존 클래스의 것을 그대로 사용
    # 클래스 객체 생성시 super() 함수를 통해 기존 클래스의 메서드 사용 가능
# class B(A):
#     def __init__():
#         super().__init__()


# Slime 만들기 -----------------
class Slime:
    def __init__(self, name='slime', hp=100):
        self.name=name
        self.hp=hp

    def acttack(self, damage):
        self.hp = self.hp - damage

# Slime 상속
class Slime_magic(Slime):
    def __init__(self, name='slime_magic', mp=50):
        super().__init__()
        self.name = name    # 메서드 오버라이딩 (Overriding)
        self.mp = mp

    def fireball(self):
        self.mp = self.mp - 10
        return 5

a = Slime_magic()
a.name
a.hp



# 국가  -----------------
class Country:      # Super Class : 상속하는 클래스
    name = '국가명'
    population = '인구'
    capital = '수도'

    def show(self):
        print('국가 클래스의 메소드입니다.')

class Korea(Country):   # Sub Class : 상속받는 클래스
    def __init__(self, name,population, capital):
        self.name = name
        self.population = population
        self.capital = capital

    def show(self):
        print(
            """
            국가의 이름은 {} 입니다.
            국가의 인구는 {} 입니다.
            국가의 수도는 {} 입니다.
            """.format(self.name, self.population, self.capital)
        )


# 연산자 오버로딩 (Operator Overloading) -------------------------------------------------------------
# https://blog.hexabrain.net/287

    # def __init__(self, arg):

    # def __call__(self):
    # def __repr__(self):           # print (official)
    # def __getitem__(self):        # index

    # def __str__(self):            # print (informal)
    # def __len__(self):            # len()
    # def __contains__(self, x):      # in
    # def __le__(self, x):        # <=
    # def __ge__(self, x):        # >=
    # def __or__(self, x):        # |
    # def __and__(self, x):       # &
    # def __next__(self):         # next()



# Iterator (Class의 Method형태) ----------------------------------------------------------------------------
# __iter__
# for문은 기본적으로 iter() 내장함수를 호출
# iter() 함수는 __next__()를 정의하는 이터레이터 객체 반환
# 남은요소가 없으면, StopIteration 예외 발생 및 for문 종료


class yrange:
    def __init__(self, n):
        self.i = 0
        self.n = n
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.i < self.n:
            i = self.i
            self.i +=1
            return i
        else:
            raise StopIteration()

y_iter = yrange(3)
next(y_iter)
next(y_iter)
next(y_iter)


for y in yrange(5):
    print(y)


# Generator (함수형태) -------------------------------------------------------------------------
# "yield"를 사용하여 데이터를 하나씩 반환하는 함수
# 이터레이터와 같은 동작 수행, 하지만 보다 간단
# __iter__()와 __next()__ 메서드 자동 생성
# 일반적인 함수처럼 작성되지만 값을 반환하고 싶을때마다 "yield" 문을 사용
# 마지막 반환 결과를 기억, 재호출시 그 위치부터 다시 시작
# 데이터가 대량일 경우 일부씩 처리시 유용
# On demand 계산을 하나씩 처리하고 싶은 경우


def gen():
    yield 1
    yield 2
    yield 3

g1 = gen()
print(type(g1))

print(next(g1))
print(next(g1))
print(next(g1))
print(next(g1))

g2 = gen()
for g in g2:
    print(g)

def square_numbers(nums):
    for x in nums:
        yield x * x

result = square_numbers([1,2,3,4,5])

for r in result:
    print(r)


# Generator Expression
numbers = [1, 2, 3, 4, 5, 6]
[x * x for x in numbers]        # [1, 4, 9, 16, 25, 36]
{x * x for x in numbers}        # {1, 4, 9, 16, 25, 36}
{x: x * x for x in numbers}        # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36}

    # generator
# * comprehension과 비슷하지만 ( ) 사용
# 제너레이터 함수의 간결한 표현 형태 하지만 융통성 부족
# 표현식만을 갖는 제너레이터 객체만 반환

(x * x for x in numbers)        # <generator object <genexpr> at 0x000001803BF142C8>

gen = (x * x for x in numbers)
# list(gen)         # [1, 4, 9, 16, 25, 36]        # __iter__ Method가 있을때에만 가능
print(next(gen))    # 1
print(next(gen))    # 4
print(next(gen))    # 9
print(next(gen))    # 16
print(next(gen))    # 25
print(next(gen))    # 36

list(gen)        # []



# First Class Object (1급 객체) -----------------------------------------------------------------------
# ○ First-class란? 아래의 조건을 만족하는 객체
    # 변수나 데이터 구조안에 담을 수 있다
    # 함수를 인수(입력값)로 전달할 수 있다
    # 함수 결과로 함수를 리턴(출력값)할 수 있다
    # (ex) 함수 = First-class object        # 변수나 데이터 구조안에 담을 수 있다.



    # Nested_Fucntion : Lexical Closures
def make_adder(n):
    def add(x):
        return x + n
    return add

plus3 = make_adder(3)
plus3(4)
plus3(10)

    # Callable
# class내 __call__ 메서드
# callable() 을 통해 해당 객체가 Callable 한지 확인할 수 있음


# Decorator @ --------------------------------------------------------------------------------
# 어떤 함수/클래스에 기능을 추가한 뒤 이를 다시 함수의 형태로 반환 (즉, 어떤 함수/클래스를 꾸며주는 함수)
# 어떤 함수의 내부 수정 없이 기능 추가시 사용

# [ 함수 데코레이터 ]
# def 데코레이터_이름(fuc):
#     def 내부함수_이름(*arg, **kwarg):
#         기존함수에 추가할 명령
#         return fuc(*arg, **kwarg)
#     return 내부함수_이름




    # Decorator 미사용시 ***
def introduce(name):
    print('my name is', name)

introduce('chaewon')


    # Decorator 사용시 ***
def decorator(func):
    def wrapper(*args ,**kwargs):   
        print('Hello')        # Decorating 할 코딩내용
        return func(*args, **kwargs)
    return wrapper

@decorator      # decorator라고 정의된 함수를 이용하겠다
def introduce_decorator(name):
    print('my name is', name)

introduce_decorator('chawon')


# [ 함수 데코레이터 ]
# def out_func(func):  # 기능을 추가할 함수를 인자로
#     def inner_func(*args, **kwargs):
#         return func(*args, **kwargs)
#     return inner_func

def decorator(func):
    def wrapper(*args, **kwargs):
       
        print('전처리')   
        print(func(*args, **kwargs))
        print('후처리')
    return wrapper

@decorator
def example():
    return '함수'
example()
# '''''''''
# 전처리
# 함수
# 후처리
# '''''''''


# [ 클래스 데코레이터 ]
# class Decorator:
#     def __init__(self, function):
#         self.function = function

#     def __call__(self, *args, **kwargs):
#         return self.function(*args, **kwargs)


class Decorator:
    def __init__(self, function):
        self.function = function

    def __call__(self, *args, **kwargs):
        print('전처리')
        print(self.function(*args, **kwargs))
        print('후처리')

@Decorator
def example():
    return '클래스'
example()
# '''''''''
# 전처리
# 클래스
# 후처리
# '''''''''








# ---------------------------------------------------------------------------------------

### [실습문제 1] 계좌 이체 메서드를 구현하세요
#### def transfer(self, other, amount):
#### 계좌에 잔고가 충분하다면 other 계좌로 송금

class Account:
    # 생성자
    def __init__(self, number, rate=0.01):
        self.number = number    # 계좌번호
        self.balance = 0        # 잔고
        self.rate = rate        # 이율

    # 메소드
    def deposit(self, amount):  # 입금
        self.balance += amount
    
    def withdraw(self, amount): # 출금
        if amount <= self.balance:
            self.balance -= amount
    
    def obtain_interest(self):
        self.balance *= (1+rate)
    
    def transfer(self, amount, account):
        if amount <= self.balance:
            self.balance -= amount
            account.deposit(amount)
            print('이체성공 : 잔액 %d' %self.balance)
        else:
            print('잔고가 부족합니다 : 잔액 %d' %self.balance)

acc1 = Account(1001)
acc2 = Account(1002)
acc3 = Account(1003)

acc1.deposit(500)
acc2.deposit(1000)

print('이체 전 잔액 ------------------------')
print('acc1.balance : %d' %acc1.balance)
print('acc2.balance : %d' %acc2.balance)

print('')
print('잔액보다 많은 금액을 이체시 ------------------------')
acc1.transfer(1000, acc2)
print('acc1.balance : %d' %acc1.balance)
print('acc2.balance : %d' %acc2.balance)

print('')
print('이체 성공 ------------------------')
acc1.transfer(300, acc2)
print('acc1.balance : %d' %acc1.balance)
print('acc2.balance : %d' %acc2.balance)



### [실습문제 2] list 클래스를 상속하면서 아래의 조건을 만족하는 Mylist 클래스를 작성하세요.
    #### list 클래스의 append 메서드를 오버라이딩 하세요.
    #### append는 리스트의 제일 처음에 원소를 추가합니다.
    #### 팁: list 클래스의 insert 메서드를 활용 Mylist 클래스에 back_append 메서드를 추가하세요.

class Mylist(list):
    def __init__(self, args):
        self.l = list(args)

    def append(self, x):
        # self.l = [x] + self.l
        self.l.insert(0, x)

    def back_append(self, x):
        # self.l += [x]
        self.l.insert(len(self.l), x)

mlist = Mylist((1,2,3))
print(mlist.l)

mlist.append(0)
print(mlist.l)

mlist.back_append(4)
print(mlist.l)

dir(Mylist)




### [실습문제 3] 주어진 문자열을 반대로 출력하는 이터레이터 클래스를 구현하세요.
class Reverse:
    def __init__(self, data):
        self.data = data
        self.index = len(data)

    def __iter__(self):
        return self

    def __next__(self):
        printing = self.data[self.index-1]
        if self.index == 0:
            raise StopIteration
        self.index -= 1
        return printing

rev = Reverse('spam')
list(rev)       # __iter__ Method가 있을때에만 가능
print(next(rev))
print(next(rev))
print(next(rev))
print(next(rev))



### [실습문제 4] 주어진 문자열을 반대로 출력하는 제너레이터 함수를 구현하세요.
def reverse_string(string):
    for i in range(len(string)):
        yield string[-1*i - 1]

for char in reverse_string('hello'):
    print(char)









#--------------------------------------------------------------------------------------------
### Python의 기본 자료구조 set 를 모방한 클래스 Set을 구현하세요
#### 임의의 원소를 중복 없이, 순서 없이 담는 집합형 자료구조
#### 아래의 함수/연산자를 명시된 기능대로 구현

class Set:
    # [과제 1Page] 
    # · 생성자 (__init__): list를 받아 중복 제거; 매개 변수 없이 생성 시, 빈 집합 상태로 생성
    # · add(elem) : Set에 elem이 존재하지 않으면 추가
    # · discard(elem) : Set에 elem이 존재하면 삭제
    # · clear() : Set에 존재하는 모든 원소 삭제
    # · __len__() : Set에 존재하는 원소 개수 반환
    def __init__(self, arg):
        self.s = []
        for a in arg:
            if a not in self.s:
                self.s.append(a)

    def add(self, x):
        if x not in self.s:
            self.s.append(x)
    
    def discard(self, x):
        if x in self.s:
            self.s.remove(x)
    
    def clear(self):
        self.s = []
    
    def __len__(self):
        return len(self.s)

    # [과제 2page]
    # __str__() : Set에 존재하는 원소를 '{1, 2, 3}'의 형태로 반환
    # __contains__(elem) : Set에 elem이 존재하면 참 반환, 아니면 거짓 반환 #### in 멤버체크
    # self <= other : self가 other의 부분집합이면 참 반환, 아니면 거짓반환
    # self >= other : other가 self의 부분집합이면 참 반환, 아니면 거짓반환
    def __str__(self):
        return '{' + ', '.join(map(str, self.s)) + '}' 

    def __contains__(self, x):
        return True if x in self.s else False

    def __le__(self, other):        # <=
        temp = []
        for o in other.s:
            if o in self.s:
                temp.append(o)
        return True if sorted(self.s) == sorted(temp) else False

    def __ge__(self, other):        # >=
        temp = []
        for s in self.s:
            if s in other.s:
                temp.append(s)
        return True if sorted(other.s) == sorted(temp) else False

    # [과제 3page]
    # self | other : self와 other의 원소를 모두 포함하는 합집합 Set반환
    # self & other : self와 other가 공통으로 포함하는 원소를 포함하는 교집합 Set 반환
    # self - other : self의 원소 중 other에 없는 원소만을 포함하는 차집합 Set 반환
    # |=, &=, -= : 위의 기능에 맞추어 구현
    def __or__(self, other):
        temp =[]
        for o in other.s:
            if o not in temp:
                temp.append(o)
        for t in self.s:
            if t not in temp:
                temp.append(t)
        result = Set(temp)
        return  result
        

    def __and__(self, other):
        temp = []
        for o in other.s:
            if o in self.s:
                temp.append(o)
        result = Set(temp)
        return result 

    def __sub__(self, other):
        temp = []
        for s in self.s:
            if s not in other.s:
                temp.append(s)
        result = Set(temp)
        return result  

    def __repr__(self):     # Print 할때 어떻게 표현하는지
        return  '{' + ', '.join(map(str, self.s)) + '}' 



a = Set([1, 2, 3, 4])
b = Set([2, 3, 4])
a
print(a)
print(b)
print()

a.discard(4)
a.discard(2)
print(a)
print(b)
print()

print(len(a))
print(1 in a)
print(1 in b)
print()

print(a | b)
print(a & b)
print(a - b)

print(a <= b)
print(a <= a | b)
print(a >= b)
print(a >= a & b)
print()

b.clear()
print(b)








#--------------------------------------------------------------------------------------------


# (6-1) 
class Thing:
    pass

example = Thing()

print(Thing)    # Class
print(example)  # Instance
print(Thing == example)


# (6-2)
class Thing2:
    letters = 'abc'         # 클래스 변수

Thing2.letters

# (6-3)
class Thing3:
    def __init__(self):
        self.letters = 'abc'    # 인스턴스 변수

something = Thing3()
something.letters


# (6-4)
class Element:
    def __init__(self, name, symbol , number):
        self.name = name
        self.symbol = symbol
        self.number = number

something4 = Element('Hydrogen', 'H', 1)
something4.name, something4.symbol, something4.number

#(6-5)
el_dict = {'name':'Hydrogen', 'symbol':'H', 'number':1}

something5 = Element(**el_dict)
something5.name, something5.symbol, something5.number


# (6-6)
class Element2:
    def __init__(self, name, symbol , number):
        self.name = name
        self.symbol = symbol
        self.number = number

    def dump(self):
        print(self.name)
        print(self.symbol)
        print(self.number)

el_dict = {'name':'Hydrogen', 'symbol':'H', 'number':1}

hydrogen = Element2(**el_dict)
hydrogen.dump()


# (6-7)
class Element3:
    def __init__(self, name, symbol , number):
        self.name = name
        self.symbol = symbol
        self.number = number

    def dump(self):
        print(self.name)
        print(self.symbol)
        print(self.number)
    
    def __str__(self):
        return self.name
el_dict = {'name':'Hydrogen', 'symbol':'H', 'number':1}

hydrogen2 = Element3(**el_dict)
hydrogen2
print(hydrogen2)


# (문제) 계산기 만들기
class Calculator:
    def __init__(self):
        self.count = {'덧셈' : 0, '뺄셈' : 0, '곱셈' : 0, '나눗셈':0}

    def Add(self, x, y):
        self.count['덧셈'] += 1
        return x + y

    def Min(self, x, y):
        self.count['뺄셈'] += 1
        return x - y
    
    def Mul(self, x, y):
        self.count['곱셈'] += 1
        return x * y

    def Div(self, x, y):
        if y !=0:
            self.count['나눗셈'] += 1
            return x / y
    
    def ShowCount(self):
        [print(i, ' : ', j) for i, j in self.count.items()]


cal = Calculator()
print('10 + 20 = %d' %cal.Add(10, 20))
print('20 - 10 = %d' %cal.Min(20, 10))
print('10 * 20 = %d' %cal.Mul(10, 10))
print('10 * 20 = %d' %cal.Mul(10, 20))
print('20 / 10 = %d' %cal.Div(10, 20))

cal.ShowCount()


# (Q1)
class Calculator2:
    def __init__(self):
        self.value = 0
    
    def add(self, x):
        self.value += x

    def minus(self, x):
        self.value -= x

cal2 = Calculator2()

cal2.add(10)
print(cal2.value)

cal2.minus(7)
print(cal2.value)


# (Q2)
class MaxlimitCalculator:
    def __init__(self):
        self.value = 0
    
    def add(self, x):
        self.value += x

        if self.value > 100:
            self.value = 100

cal3 = MaxlimitCalculator()
print(cal3.value)

cal3.add(50)
print(cal3.value)

cal3.add(60)
print(cal3.value)
