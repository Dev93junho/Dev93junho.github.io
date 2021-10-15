#include <bits/stdc++.h>

using namespace std;

bool visited[9];
vector<int> graph[9];

void dfs(int x){
    visited[x] = true;
    cout << x <<' ';
    for (int i = 0; i < graph[x].size(); i++){
        int y = graph[x][i];
        if (!visited[y]) dfs(y); 
    }
}

int main(void) {
    /*
    */
   return 0;
}
    
