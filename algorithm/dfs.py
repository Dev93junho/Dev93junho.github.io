"""
그래프 준비
번호가 낮은 인접 노드부터 방문
"""

def dfs(graph, v, visited):
    #현재노드 방문처리
    visited[v] = True
    print(v, end=' ')
    #현재 노드와 연결된 다른 노드를 재귀적으로 방문
    for i in graph[v]:
        if not visited[i]:
            dfs(graph, i, visited)

graph =[]

visited = [False] * 9

dfs(graph, 1, visited)