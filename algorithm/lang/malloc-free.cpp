#include <iostream>
#include <string.h>

int main(void) {
    float *a = (float *)malloc(sizeof(float) *5);
    float *b = a + 2;
    free(a);

    return 0;
}