from collections import deque

#define bfs methods
def bfs(graph, start, visited):
    #using deque Lib for implement Queue
    queue = deque([start])
    visited[start] = True #현재 노드를 방문처리
    # 큐가 빌때 까지 반복
    while queue:
        v = queue.popleft()
        print(v, end='')
        # 아직 방문하지 않은 인접 원소들을 큐에 삽입
        for i in graph[v]:
            queue.append(i)
            visited[i] = True

#not fully index
graph = [
    [],
    [1,2,3]
]

visited = [False] * 9

bfs(graph, 1, visited)