#include <bits/stdc++.h>

using namespace std;

queue<int> q;

int maain(void){
    q.push(1);
    q.push(3);
    q.push(4);
    q.push(5);
    q.pop();
    //가장 먼저 들어온 원소부터 추출(FIFO)
    while (!q.empty()) {
        cout<<q.front()<<' ';
        q.pop();
    }
}