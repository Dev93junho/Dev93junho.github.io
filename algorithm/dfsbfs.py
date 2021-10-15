'''
DATA STRUCT
stack : FILO

'''
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

# 인접행렬 방식 예제
INF = 999999 # 무한의 비용

graph = [
  [0, 7, 5],
  [7, 0, INF],
  [5, INF, 0]
]

print(graph)

