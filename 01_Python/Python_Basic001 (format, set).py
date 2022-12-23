
# local
# http://127.0.0.1:9988/?token=7f53ed490e7e4fdd3f424825ec9bf7221242d2187e08a037

odd_list = list(range(1, 10, 2))
even_list = list(range(0, 10, 2))

odd_list + even_list

# 할일 ---------------------------------------------------------------------------------------------
# Format --------------------------------------------------------------------
format(123.45, '.5f')
# format(123.45, 's')        # Error
format('123.45', 's')
format(123.45, '5d')        # Error
format(int(123.45), '5d')
format(int(123.45), '05d')

format(123, '.5f')
format(123, '5d')

print('%d %% abc' %5)




# set(집합) *** ------------------------------------------------------------------------------
s = {1, 2, 3, 4, 5, 3, 4}
s
# s[0]      # Error : 인덱싱 불가

s.add(10)               # 원소한개 추가
print(s)

s.remove(2)             # 삭제 (없는원소를 제거하라는 명령에는 에러 발생)
print(s)

s.update([1,3,5,7,9])   # 여러개 원소 추가
print(s)

s.discard(7)            # 삭제 (없는원소를 제거하라는 명령에도 에러는 미발생)
print(s)

s.clear()               # 원소 모두 삭제
print(s)

    # 합집합, 교집합, 차집합
s1 = {1, 2, 3}
s2 = {3, 4, 5}

print(s1 | s2)              # 합집합
print(s1 & s2)              # 교집합
print(s1 - s2)              # 차집합
# print(s1 + s2)            # Error

print(s1.union(s2))         # 합집합
print(s1.intersection(s2))  # 교집합


