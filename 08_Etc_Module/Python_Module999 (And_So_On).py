
# pip에 자신이 만든 모듈 등록하는 방법
# https://m.blog.naver.com/cyydo96/221585902894

# itertools ---------------------------------------------------------
from itertools import combinations, chain, product      # ,islice, count

a = [1,2,3,4]
b = [5,6,7]
c = [8, 9]

# combinations
iter1 = combinations(a, 2)
for i1 in iter1:
    print(i1)

# combinations
iter2 = chain(a)
next(iter2)     # 1
next(iter2)     # 2
next(iter2)     # 3
next(iter2)     # 4

# product
iter3 = product(a,b)
for i3 in iter3:
    print(i3)



# [ Missingpy Library ]  -----------------------------------------------------------------------

from missingpy import MissForest        #  Randomforest 방식으로 결측치를 채우는 Library
# https://pypi.org/project/missingpy/
# imputer = MissForest()
# X_imputed = imputer.fit_transform(X)
