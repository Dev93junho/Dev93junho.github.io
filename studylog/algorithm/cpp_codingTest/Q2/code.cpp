#include <iostream>
using namespace std;
int main(int argc, char* argv[]){
	freopen(argv[1], "rt", stdin);
	freopen(argv[2], "w", stdout);    

    int a, b, i, sum=0;
    cin >> a >> b;
    for(i=a; i<b; i++){
        cout<<i<<"+";
        sum=sum+i;
    }
    cout<<b<<"=";
    cout<<sum+b+i;
    return 0;
}