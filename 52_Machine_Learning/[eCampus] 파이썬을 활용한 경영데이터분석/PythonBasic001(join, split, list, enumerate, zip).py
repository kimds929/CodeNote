
# 【 pythonic code 】
# join, split
colors =['red', 'blue', 'green']
print('.'.join(colors))
print('\n'.join(colors))
print('\t'.join(colors))

strings = 'http://www.naver.com'
strings.split('.')

# list comprehension, 
[i+10 for i in range(10) ]

text_1 = 'Hello'
text_2 = 'World'
[a + b for a in text_1 for b in text_2]
[[a + b for a in text_1] for b in text_2]
[a + b for a in text_1 for b in text_2 if a==b]
[[a + b for a in text_1 if a==b] for b in text_2]

[b+'a' for a in colors for b in a]

text_3 = 'CBA'
text_4 = 'FGA'
[a+b for a in text_3 for b in text_4]
[a+b for a in text_3 for b in text_4].sort()

# enumerate : list타입에서 주로사용하는 기법으로 인덱스와 벨류값을 함께추출하는 기법
test = 'abcde'
list(enumerate(test))
list(enumerate(colors))

test2 = "Hello World Python"
ohe = test2.split()
for i, v in enumerate(ohe):
    print(i)
    print(v)

# zip : 같은 길이의 list 2개가 있을떄, 같은 위치에 있는 값들을 추출하는 기법
a = [1, 2, 3]
b = [10, 20, 30]
for c in zip(a,b):
    print(c)

[sum(x) for x in zip((1,2,3), (10,20,30), (100,200,300))]


alist= ['a1', 'a2', 'a3']
blist = ['b1', 'b2', 'b3']
for c in enumerate(zip(alist, blist)):
    print(c)


