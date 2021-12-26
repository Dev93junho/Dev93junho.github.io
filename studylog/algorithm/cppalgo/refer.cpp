#include <iostream>

using namespace std;

int main(void)
{
    int num1 = 20;
    int& num2 = num1; // &은 num1과 같은 메모리 공간을 가리키게함

    // 참조자는 변수에 대해서만 선언이 가능, 선언과 동시에 참조할 변수로 초기화
    int* num3 = &num1; // num1의 시작주소를 가리키는 포인터

    cout << "num1의 주소: "<< &num1<< endl;
    cout << "num2의 주소: "<< &num2 << endl;
    cout << "num3의 주소: "<< &num3 << endl;

    cout << "sizeof num1: "<< sizeof(num1) << endl;
    cout << "sizeof num2: "<< sizeof(num2) << endl;
    cout << "sizeof num3: "<< sizeof(num3) << endl;

    num2 = 3047; // num2에 값을 할당 

    cout << "num1 = " << num1 << endl;
    cout << "num2 = " << num2 << endl;
    cout << "num3 = " << num3 << endl;

    return 0;
}