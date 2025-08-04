import pandas as pd
import numpy as np

# 데이터의 분리(Splitting Data)

# 머신 러닝 모델을 학습시키고 평가하기 위해서는 데이터를 적절하게 분리하는 작업이 필요합니다.
# 대부분의 경우에서 지도 학습(Supervised Learning)을 다루는데, 이번에는 지도 학습을 위한 데이터 분리 작업에 대해서 배웁니다.


"""

지도 학습(Supervised Learning)
-> 정답이 적혀져 있는 문제지를 문제와 정답을 함께 보면서 공부, 
향후에 정답이 없는 문제에서도 정답을 예측할 수 있도록 하는 학습 방법


<훈련 데이터>
X_train : 문제지 데이터
y_train : 문제지에 대한 정답 데이터

<테스트 데이터>
x_test : 시험지 테스트
y_test : 시험지에 대한 정답 데이터

머신은 x_train과 y_train에 대해서 먼저 학습 -> x_train와 y_train사이에서 규칙을 도출하면서 정리해 나감. 
정리가 끝나고 학습이 종료되면, y_test는 보여주지 않고, x_test에 대해서 정답을 예측하게 함 -> 예측 결과가 실제 정답인 y_test를 비교하면서
정답을 얼마나 맞췄는지 평가함. 이를 "정확도(Accuracy)"임.

"""

# 분리해보기(Zip함수)

X, y = zip(['a', 1], ['b', 2], ['c', 3])
print('X 데이터 :',X)
print('y 데이터 :',y)
# 첫번재 등장한 원소끼리, 두번째 등장한 원소끼리 묶인 것을 확인 가능

# 분리해보기(데이터프레임 이용)
values = [['당신에게 드리는 마지막 혜택!', 1],
['내일 뵐 수 있을지 확인 부탁드...', 0],
['도연씨. 잘 지내시죠? 오랜만입...', 0],
['(광고) AI로 주가를 예측할 수 있다!', 1]]
columns = ['메일 본문', '                    스팸 메일 유무']

df = pd.DataFrame(values, columns=columns)
print(df)


# 데이터프레임은 열의 이름으로 각 열에 접근이 가능함. 손쉽게 X데이터와 Y데이터를 분리할 수 있음
X = df['메일 본문']
Y = df['                    스팸 메일 유무']
print(X.to_list())
print(Y.to_list())

# 분리하기(Numpy를 이용하여 분리하기)
np_array = np.arange(0,16).reshape((4,4))
print('전체 데이터 :')
print(np_array)

X = np_array[:, :3]
# : <- 전체의 의미임 그래서, 행전체, 열 0~2까지
y = np_array[:,3]
# 행전체, 열 인덱스 3만 가져옴 0 1 2 3

print('X 데이터 :')
print(X)
print('y 데이터 :',y)



# 테스트 데이터 분리하기
# 1) 사이킷 런을 이용하여 분리하기
"""

사이킷런은 학습용 테스트와 테스트용 데이터를 쉽게 분리할 수 있게 해주는 train_test_split()를 지원
::
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=1234)

X : 독립 변수 데이터. (배열이나 데이터프레임)
y : 종속 변수 데이터. 레이블 데이터.
test_size : 테스트용 데이터 개수를 지정한다. 1보다 작은 실수를 기재할 경우, 비율을 나타낸다.
train_size : 학습용 데이터의 개수를 지정한다. 1보다 작은 실수를 기재할 경우, 비율을 나타낸다.
random_state : 난수 시드
::

"""



# 2) 수동으로 분리하기
# 임의 X데이터와 y데이터 생성
X, y = np.arange(0,24).reshape((12,2)), range(12)

print('X 전체 데이터 :')
print(X)
print('y 전체 데이터 :')
print(list(y))

""" 출력결과
X 전체 데이터 :
[[ 0  1]
 [ 2  3]
 [ 4  5]
 [ 6  7]
 [ 8  9]
 [10 11]
 [12 13]
 [14 15]
 [16 17]
 [18 19]
 [20 21]
 [22 23]]
y 전체 데이터 :
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

"""

# num_of_train 훈련 데이터의 개수
# num_of_test 테스트 데이터의 개수
num_of_train = int(len(X) * 0.8) # 데이터의 전체 길이의 80%에 해당하는 길이값을 구한다.
num_of_test = int(len(X) - num_of_train) # 전체 길이에서 80%에 해당하는 길이를 뺀다.
print('훈련 데이터의 크기 :',num_of_train)
print('테스트 데이터의 크기 :',num_of_test)
"""
출력결과
훈련 데이터의 크기 : 9
테스트 데이터의 크기 : 3
"""

X_test = X[num_of_train:] # 전체 데이터 중에서 20%만큼 뒤의 데이터 저장
y_test = y[num_of_train:] # 전체 데이터 중에서 20%만큼 뒤의 데이터 저장
X_train = X[:num_of_train] # 전체 데이터 중에서 80%만큼 앞의 데이터 저장
y_train = y[:num_of_train] # 전체 데이터 중에서 80%만큼 앞의 데이터 저장

print('X 테스트 데이터 :')
print(X_test)
print('y 테스트 데이터 :')
print(list(y_test))

# 출력결과
X 테스트 데이터 :
[[18 19]
 [20 21]
 [22 23]]
y 테스트 데이터 :
[9, 10, 11]
