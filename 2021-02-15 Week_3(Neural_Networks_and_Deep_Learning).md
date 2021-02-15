# 3주차

## Neural Networks and Deep Learning

### Activation functions

시그모이드 같은 함수를 활성함수라고 한다. 활성 함수는 이외에도 tanh 등이 있다.  tanh는 대부분의 학습기에서 좋은 성능을 내는데, 그 이유는 데이터를 중심에 위치시키는 효과가 있기 때문에 데이터 평균 값이 0에 가깝게 되며 이렇게 함으로써 다음 층에서의 학습이 더 잘 되기 때문이다.

  예외적으로 시그모이드를 사용하는 경우는 '출력층'인 경우이다. y(출력값)는 0이거나 1인 경우 y의 예측 값 또한 0과 1로 맞춰야 하기 때문이다.

 시그모이드나 tanh의 단점은 z 값이 매우 크거나 작을 때 함수의 기울기가 매우 작은 값이 되므로 학습이 잘 되지 않는다.(vanishing gradient). 따라서 **RELU** 함수를 대신 사용하게 된다. z값이 0 이하인 부분에서 기술적으로는 미분값을 정의하기 어렵지만, 구현을 컴퓨터로 하므로 0.0000000.... 등의 매우 작은 값을 얻게 할 수 있다.

 활성함수를 결정할 때 한가지 규칙이 있는데, 바로 **이진 분류 문제에서 출력 값이 0과 1일 때에는 시그모이드 활성 함수가 적절한 선택이라는 것과, 다른 층에서는 RELU가 활성함수를 선택하는 보편적인 기준이 될 것이라는 점이다.** RELU의 한가지 단점은 0 이하에서 미분 값이 0이 된다는 것이다. 이를 보완하기 위하여 **leaky RELU**라는 활성함수를 사용하게 된다. RELU와 leaky RELU의 장점은 z의 많은 영역에서의 활성함수의 미분 값이 0과 매우 다른값을 갖게 해준다는 것이다. 이들은 학습을 더욱 빠르게 할 수 있도록 도와준다. z값의 절반은 0이므로 미분 값 또한 0에 가깝지만. 실제로는 은닉층 유닛들의 z 값이 0보다는 충분히 클 것이므로 걱정할 필요가 없다.

![3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%205d06a15045924738926b1cdead7ec64c/Untitled.png](3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%205d06a15045924738926b1cdead7ec64c/Untitled.png)

## Why do you need non-linear activation functions?

  만일 활성 함수를 identity activation function으로 사용할 때 신경망은 선형함수에서 입력 값에 대한 결과 값을 주게 되는 것이다. 이때 은닉층은 쓸모 없게 된다. 2개의 선형 함수의 구성 요소는 그 자체가 선형 함수이기 때문이다. 예외적으로  identity activation function를 적용하는 경우는 출력층에서의 y가 실수 값인 경우 선형 함수를 사용할 수 있다. 물론 이 경우에도 은닉 층에 다른 활성함수를 적용하지 않는다.

## Derivatives of activation functions

![3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%205d06a15045924738926b1cdead7ec64c/Untitled%201.png](3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%205d06a15045924738926b1cdead7ec64c/Untitled%201.png)

![3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%205d06a15045924738926b1cdead7ec64c/Untitled%202.png](3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%205d06a15045924738926b1cdead7ec64c/Untitled%202.png)

![3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%205d06a15045924738926b1cdead7ec64c/Untitled%203.png](3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%205d06a15045924738926b1cdead7ec64c/Untitled%203.png)

![3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%205d06a15045924738926b1cdead7ec64c/Untitled%204.png](3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%205d06a15045924738926b1cdead7ec64c/Untitled%204.png)

![3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%205d06a15045924738926b1cdead7ec64c/Untitled%205.png](3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%205d06a15045924738926b1cdead7ec64c/Untitled%205.png)

![3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%205d06a15045924738926b1cdead7ec64c/Untitled%206.png](3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%205d06a15045924738926b1cdead7ec64c/Untitled%206.png)

![3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%205d06a15045924738926b1cdead7ec64c/Untitled%207.png](3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%205d06a15045924738926b1cdead7ec64c/Untitled%207.png)

![3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%205d06a15045924738926b1cdead7ec64c/Untitled%208.png](3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%205d06a15045924738926b1cdead7ec64c/Untitled%208.png)

![3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%205d06a15045924738926b1cdead7ec64c/Untitled%209.png](3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%205d06a15045924738926b1cdead7ec64c/Untitled%209.png)

## Gradient descent for Neural Networks

## Formulas for computing drivatives

![3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%205d06a15045924738926b1cdead7ec64c/Untitled%2010.png](3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%205d06a15045924738926b1cdead7ec64c/Untitled%2010.png)

![3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%205d06a15045924738926b1cdead7ec64c/Untitled%2011.png](3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%205d06a15045924738926b1cdead7ec64c/Untitled%2011.png)

## Random Initialization - 초기화가 중요한 이유

![3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%205d06a15045924738926b1cdead7ec64c/Untitled%2012.png](3%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%205d06a15045924738926b1cdead7ec64c/Untitled%2012.png)