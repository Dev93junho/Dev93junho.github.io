#include <iostream>

using namespace std; 
/*
namespace 기반 함수선언 및 정의 구분
중첩되는 함수 선언시 namespace를 잘활용할 수 있다
using을 이용해 이름공간을 명시 할 수 있다
*/

int main(void){
    int num = 10;
    int i = 0;

    cout << "true: " << true << endl;
    cout << "false: " << false << endl; 

    while(true)
    {
        cout << i++ << ' ';
        if(i > num)
        {
            break;
        }
    }

    cout << endl;

    cout << "size of 1: " << sizeof(1) << endl;
    cout << "size of 0: " << sizeof(0) << endl;
    cout << "size of true" << sizeof(true) << endl;
    cout << "size of false" << sizeof(false) << endl;

    system("pause"); // VC에서만 필요한 기능

    return 0;
}