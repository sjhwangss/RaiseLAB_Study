# What is a Tensor?

A **tensor** represents algebraic relationships, defined by **dependent bases**, comparing the relationships among elements. Generally, tensors are described by the **number of axes (dimensions)** they have.

---

## Dimensions of a Tensor

- $n=0$: Tensor with 0 axes, known as a **scalar**.
- $n=1$: Tensor with 1 axis, known as a **vector**.
- $n=2$: Tensor with 2 axes, known as a **matrix**.
- $n=N$: Tensor with $N$ axes. For $N \geq 3$, tensors typically do not have specific names but are conceptually valid, and their axes can extend indefinitely.

---

## Common Misconception

**Is the number of vectors equal to the tensor's dimensions?**  
Not exactly. The **number of axes (dimensions)** determines a tensor's rank, not the number of vectors.

---

# Python Module: PyTorch Tensor Functions

### 1. Tensor Creation
- `torch.arange(start, end, step)`: Creates a 1D tensor (vector) with values from `start` to `end`, incremented by `step`.
  
- `torch.zeros(dimensions)`, `torch.ones(dimensions)`: Generates tensors filled with zeros or ones.

- `torch.randn(dimensions)`: Generates a tensor with **random values** following a Gaussian distribution.
  
  ![Gaussian Distribution](https://i.imgur.com/WjUylXj.png)

- `torch.tensor(list)`: Converts a Python list into a tensor.  
  - Providing a single number creates a **scalar tensor** with 0 axes.

---

### 2. Tensor Properties
- `torch.shape`: Returns the shape (dimensions) of a tensor.
- `torch.reshape(dimensions)`: Reshapes a tensor to new dimensions (e.g., vector → matrix).

---

# Linear Algebra with Tensors

---

## Basic Concepts of Tensors

- **Scalar:** 0 axes
- **Vector:** 1 axis
- **Matrix:** 2 axes

---

## Transpose of a Matrix

- **Definition:** The transpose of a matrix swaps its **rows and columns**, denoted as $A^T$.
- **Python Implementation:** The transpose of tensor `A` is expressed as `A.T`.

---

## Operations

1. **Element-wise Multiplication:**
   - Syntax: `A * B`

2. **Matrix Multiplication:**
   - Syntax: `A @ B`

It is important to distinguish between element-wise and matrix multiplications.

---

In the next chapter, we will explore **Numerical Differentiation** in detail.

---

# 텐서란? (What is a Tensor?)

텐서는 대수적인 관계를 나타내는 개념으로, **종속적인 기저**를 바탕으로 정의된다. 텐서는 원소들 간의 기저 관계를 비교하며, 일반적으로 **차원의 축(Axis)**의 개수로 설명할 수 있다.

---

## 차원의 축 (Dimensions)

- $n=0$: 축의 개수가 0개인 텐서. **스칼라(Scalar)**라고 불린다.
- $n=1$: 축의 개수가 1개인 텐서. **벡터(Vector)**라고 불린다.
- $n=2$: 축의 개수가 2개인 텐서. **행렬(Matrix)**이라고 불린다.
- $n=N$: 축의 개수가 $N$개인 텐서. 3 이상의 텐서는 별도의 이름이 없지만, 개념적으로 존재하며, 축의 개수는 무한대로 확장 가능하다.

---

## 흔한 착각

**벡터의 개수가 텐서의 차원을 나타내는가?**  
엄밀히 말하면, 텐서의 차원은 **축(axis)의 개수**로 결정된다. 벡터의 개수와 텐서의 차원을 혼동하지 말아야 한다.

---

# Python 모듈: PyTorch 관련 함수

### 1. 텐서 생성
- `torch.arange(start, end, step)`: 시작값, 끝값, 스텝 크기에 따라 1차원 텐서를 생성.
  - 차원을 명시하지 않으면 **1차원 텐서(벡터)**가 생성된다.
  
- `torch.zeros(dimensions)`, `torch.ones(dimensions)`: 모든 값이 0 또는 1인 텐서를 생성.

- `torch.randn(dimensions)`: 주어진 차원에 따라, **가우시안 분포**를 따르는 랜덤 값으로 텐서를 생성.
  
  ![Gaussian Distribution](https://i.imgur.com/WjUylXj.png)

- `torch.tensor(list)`: Python 리스트를 텐서로 변환.  
  - 리스트 대신 단일 숫자를 넣으면, 축이 $0$인 **스칼라** 텐서가 생성된다.

---

### 2. 텐서의 속성
- `torch.shape`: 텐서의 차원 정보 반환.
- `torch.reshape(dimensions)`: 텐서의 차원을 변경. (예: 벡터 → 행렬, 행렬 → 벡터)

---

# 선형대수 (Linear Algebra)

---

## 텐서의 기본 개념

- **스칼라 (Scalar):** 축의 개수 0
- **벡터 (Vector):** 축의 개수 1
- **행렬 (Matrix):** 축의 개수 2

---

## 전치 행렬 (Transpose)

- **정의:** 행렬에서 **행과 열을 교환**한 행렬을 **전치 행렬**이라 하며, $A^T$로 표기.
- **Python 구현:** 텐서 변수 `A`의 전치는 `A.T`로 표현.

---

## 연산

1. **원소별 곱(Element-wise Multiplication):**
   - 연산: `A * B`

2. **행렬 곱(Matrix Multiplication):**
   - 연산: `A @ B`

행렬 곱과 원소별 곱은 명확히 구분하여 사용해야 한다.

---

다음 챕터에서는 **수치 미분(Numerical Differentiation)**에 대해 다룰 것이다.


