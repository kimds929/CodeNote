import torch

################################################################
# (dot, matmul, bmm)
# https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=cjh226&logNo=221356884894
# (*) numpy.dot(a, b, out=None) : Dot product of two arrays. (내적곱)
#   . 텐서(또는 고차원 배열)의 곱연산
#   . 5) 만약 a가 N차원 배열이고 b가 2이상의 M차원 배열이라면, 
#        dot(a,b)는 a의 마지막 축과 b의 뒤에서 두번째 축과의 내적으로 계산된다.
# np.dot(A,B)[i,j,k,m] == np.sum(A[i,j,:] * B[k,:,m]) → (i, j, k, m)

# (*) numpy.matmul(a, b, out=None) : Matrix product of two arrays. (행렬곱)
#   . 두번째 설명이 고차원 배열(N>2)에 대한 내용
#   . 2) 만약 배열이 2차원보다 클 경우, 
#        마지막 2개의 축으로 이루어진 행렬을 나머지 축에 따라 쌓아놓은 것이라고 생각한다.
# np.matmul(A,B)[i,j,k] == np.sum(A[i,j,:] * B[i,:,k]) → i, j, k


# (*) torch.bmm : matrix multiplicatoin
#  . bmm(A,B) [i,k] == np.sum(A[i,j], b[j,k]) → i, k

A0 = np.arange(2*3*4).reshape((2,3,4))

B1 = np.arange(2*3*4).reshape((2,3,4))
B2 = np.arange(2*3*4).reshape((2,4,3))
B3 = np.arange(2*3*4).reshape((3,2,4))
B4 = np.arange(2*3*4).reshape((3,4,2))
B5 = np.arange(2*3*4).reshape((4,2,3))
B6 = np.arange(2*3*4).reshape((4,3,2))

# (dot) operation
A0.shape    # (2,3,4)
np.dot(A0, B1) # (2,3,4) Error
np.dot(A0, B2) # (2,4,3) Ok -> (2,3,2,3)
np.dot(A0, B3) # (3,2,4) Error
np.dot(A0, B4) # (3,4,2) Ok -> (2,3,3,2)
np.dot(A0, B5) # (4,2,3) Error
np.dot(A0, B6) # (4,3,2) Error


# (matmul) opertaion
A0.shape    # (2,3,4)
np.matmul(A0, B1) # (2,3,4) Error
np.matmul(A0, B2) # (2,4,3) Ok -> (2,3,3)
np.matmul(A0, B3) # (3,2,4) Error
np.matmul(A0, B4) # (3,4,2) Error
np.matmul(A0, B5) # (4,2,3) Error
np.matmul(A0, B6) # (4,3,2) Error




################################################################
# (tril) 아래 대각 행렬을 만들어줌
ten = torch.rand(5,5)
torch.tril(ten)


###################################################################
# (expand_as) 차원을 맞춰줌 
ten = torch.rand(10,3,2)

ten1 = torch.arange(0, 10)
ten1.unsqueeze(-1).unsqueeze(-1).expand_as(ten).shape

ten2 = torch.arange(0, 3)
ten2.unsqueeze(0).unsqueeze(-1).expand_as(ten).shape

ten3 = torch.arange(0, 2)
ten3.unsqueeze(0).unsqueeze(0).expand_as(ten).shape
##################################################################