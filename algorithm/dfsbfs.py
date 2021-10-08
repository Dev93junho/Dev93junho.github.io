# 인접행렬 방식 예제
INF = 999999 # 무한의 비용

graph = [
  [0, 7, 5],
  [7, 0, INF],
  [5, INF, 0]
]

print(graph)


# 인접리스트 방식 예제
graph [[] for _ in range(3)]

graph[0].append((1,7))
graph[0].append((2,3))

graph[1].append((0,7))

graph[2].append((0,5))

print(graph)

# dfs 예제
def dfs (graph, v, visited):
  #현재 노드를 방분 처리
  visited[v] = True
  print(v, end='')
  #현재 노드와 연결된 다른 노드를 재귀적으로 방문
  for i in graph[v]:
    if not visitied[i]:
      dfs(graph, v, visited)

graph = [
  [],
  [2,3,8],
  [1,7],
]