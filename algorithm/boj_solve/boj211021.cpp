// 2780

#include <iostream>
using namespace std;
int main(void)
{
    int N, temp;
    cin>>N; // input data
    int arr[N];
    // Load data
    for (int i=0;i<N;i++)
    {
        cin>>arr[i];
    }
    
    // 첫원소부터 정렬시작
    for (int j = 0;j<N-1;j++)
        for (int k = j+1; k<N;k++)
    {
        if(arr[j]>arr[k]) // arr[j] > arr[k] 인 경우 둘 위치를 swap
        {
            temp = arr[k];
            arr[k] = arr[j];
            arr[j] = temp;                
        }
    }
    for(int a=0; a<N; a++)
    {
        if (a+1<N && arr[a] == arr[a+1]) // 원소가 같은 경우 출력 X
            continue;
        cout << arr[a] << endl;
    }
}

//
