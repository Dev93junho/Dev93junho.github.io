#선택정렬
'''
처리되지 않은 데이터 중에서 가장 작은 데이터를 선택해 맨앞에 있는 데이터와 바꾸는 것을 반복함
기준 데이터를 잡고 가장작은 원소와 교체를 반복함
처리되지 않은 나머지 배열은 선형탐색과정을 거쳐 정렬되어짐
'''

array = [7, 5, 1, 0, 9, 4, 2, 3]

for i in range(len(array)):
    min_index = i
    for j in range(i + 1, len(array)):
        if array[min_index] > array[j]:
            min_index =j
    array[i], array[min_index] = array[min_index], array[i] # swap

print(array)

# 삽입 정렬
'''
처리되지 않은 데이터를 하나씩 골라 적절한 위치에 삽임함
선택 정렬에 비해 구현 난이도가 높지만, 일반적으로 더 효율적임
삽입 정렬의 시간 복잡도는 O(N^2) 이며, 선택정렬과 마찬가지로 반복문이 두번 중첩되어 사용됨
삽입정렬은 현재 리스트이 데이터가 거의 정렬되어 있는 상탤면 매우 빠르게 동작할 수 있음
최선의 경우 O(N)의 시간 복잡도를 가짐
'''
for a in range(1, len(array)):
    for j in range(i, 0, -1): #인덱스 i부터 1까지 1씩 감소하며 반복하는 문법
        if array[j] < array[j-1]:
            arrya[j], array[j -1] = array[j -1], array[j] # swap
        else:
            break # 자기보다 작은 데이터를 만나면 멈춤

print(array)

# 퀵 정렬
'''
가장 많이 사용하는 정렬 알고리즘, 정렬라이브러리의 베이스가 됨
기준데이터(피봇)을 설정하고 기준보다 큰데이터와 작은데이터의 위치를 바꾸는 방법
왼쪽에서 부터 피봇보다 큰값을 선택하고, 오른쪽에서 부터는 피봇보다 작은 것을을 정렬하기
basic 퀵정렬은 첫번째 데이터를 pivot으로 설정함
피봇을 기준으로 데이터 묶음을 나누는 작업을 분할(Divide, partition)이라고함
각 사이드별로 다시 퀵정렬을 재귀실행함
이상적인 경우 분할이 절반씩 일어난다면 전체 연산횟수로 O(NlogN)을 기대할 수 있음, 
최악의 경우에는 O(N^2)일수 있음> 거의 편향적인 피봇이 만들어지는 경우
'''

def quick_sort(array, start, end):
    if start >= end:
        return # 원소가 1개인 경우 종료
    pivot = start # 피봇은 첫 번재 원소
    left = start + 1 # 0번째 원소를 pivot으로 설정했으므로 그 다음원소부터 시작
    right = end 
    while (left <= right):
        #피봇보다 큰 데이터를 찾을 때까지 반복
        while(left <= end and array[left] <= array[pivot]):
            left += 1 # 왼쪽부터 index 증가
        # 피봇보다 작은 데이터를 찾을 때까지 반복
        while(right > start and array[right] >= array[pivot]):
            right -= 1 # 오른쪽부터 index 감소
        if(left > right): # 왼쪽이 오른쪽보다 index가 커저셔 교차하게 되는 경우 pivot을 교체
            array[right], array[pivot] = array[pivot], array[right]
        else:
            array[left], array[pivot] = array[pivot], array[left]
    quick_sort(array, start, right -1)
    quick_sort(array, right + 1, end)

quick_sort(array, 0, len(array) - 1)
print(array)


