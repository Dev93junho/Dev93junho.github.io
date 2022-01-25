#include <iostream>
using namespace std;
int main(int argc, char* argv[]){
	freopen(argv[1], "rt", stdin);
	freopen(argv[2], "w", stdout);
    int n, m, i, sum=0;
    cin >>n>>m;
    for(i=1; i<=n; i++){
        if(i%m==0){
            sum=sum+i;
        }
    }
    cout << sum;
    return 0;
}