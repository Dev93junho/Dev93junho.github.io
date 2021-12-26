#include <iostream>
#include <string.h>

int main(void) {
    float *a = (float *)malloc(sizeof(float) *5);
    float *b = a + 2;
    free(a);

    return 0;
}
/*
- malloc은 memory를 확보하는 C lang 함수
- free는 메모리에 할당된 데이터를 메모리에서 해제함
- 사용된 메로리 공간을 비우지 않으면 메모리 누수현상 발생함 
- 최근에는 new와 delete로 대체하여 사용한다
*/