'''
DATA STRUCT
stack : FILO
queue : FIFO (like tunnel)

'''
# Stack
stack = []

stack.append(5)
stack.append(2)
stack.append(3)
stack.pop()
stack.append(7)
stack.append(1)
stack.append(4)
stack.pop()

print(stack[::-1])
print(stack)


#Queue
from collections import deque
queue = deque()

queue.append(1)
queue.append(2)
queue.append(3)
queue.append(4)
queue.append(5)
queue.append(6)
queue.popleft() #가장왼쪽에 있는것을 삭제, 관행임

print(queue)
queue.reverse() #역순전환
print(queue)

#reculsive
'''
재귀함수의 종료조건을 반드시 넣어줄 것
'''
# 유클리드 호제법
def gcd(a,b):
    if a % b == 0:
      return b
    else:
      return gcd(b, a % b)
print(gcd(192, 162))

# 인접행렬 방식 예제
INF = 999999 # 무한의 비용

graph = [
  [0, 7, 5],
  [7, 0, INF],
  [5, INF, 0]
]

print(graph)

