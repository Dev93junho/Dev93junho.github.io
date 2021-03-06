// 2750

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

//2751 : vector, algorithm header 사용
/*
개인적인 소견으로 공부하면서 외부 라이브러리는
최대한 지양해야할 방향 생각됨.
특히 algorithm 라이브러리의 경우 실력향상에 방해될 것 같음
c++ Standard Template Library(STL)은 극히 제한하자
vector : 메모리 heap에 생성되어 동적할당되어짐
array에 비해 성능이 떨어지나 메모리를 효율적으로 관리하고 예외처리가 쉬움
*/
#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;
int main(){
    int n, temp;
    vector <int> a;
    cin >> n;
    for(int i =0; i < n; i++){
        cin >> temp;
        a.push_back(temp);
    }
    sort(a.begin(), a.end());
    for (int i = 0; i < n; i++){
        cout << a[i] << '\n';
    }
}

// 1427 : sort inside
/*
방법을 두가지 써보자
m1 : algorithm 라이브러리를 사용한 경우
m2 : iostream과 같은 기본 라이브러리만 사용한 경우 
*/

//m1. algorithm 라이브러리를 사용한 경우
// 이 경우 sort 함수를 이용한다
#include <iostream>
#include <algorithm>
#include <cstring>
using namespace std;
char arr[10];
int main(void){
    cin>>arr;

    sort(arr, arr+strlen(arr), greater<int>());
    for (int i=0; i<strlen(arr); i++){
        cout <<arr[i];
    }

    return 0;
}


//m2. iostream 만 사용한 경우
//Bubble Sort?


//2839
#include <iostream>
using namespace std;
int main(void){
    int N, m, remainder, quotient;
    cin >>N;
    m = N/5; // 5kg 봉지 최대 개수
    while(m >=0){
        //initialize variable
        remainder = 0;
        quotient = 0;
        if(m > 0)
        {
            remainder = N - 5*m;
            quotient = m;
        }
        else
            remainder = N;
        // 3kg 봉지 개수
        quotient += remainder / 3;
        remainder = remainder % 3;
        
        if( remainder == 0)
        {
            cout << quotient;
            break;
        }
        m--; // 나누어 떨어지지 않을 경우 5kg 봉지 개수를 줄임        
    }
    if(remainder !=0)
        cout << -1;
}