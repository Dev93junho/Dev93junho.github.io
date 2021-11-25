#include <iostream>

using namespace std;

int main(void){
    int num = 10;
    int i = 0;

    cout << "ture: " << true << endl;
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