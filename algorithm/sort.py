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
'''
for a in range(1, len(array)):
    for j in range(i, 0, -1): #인덱스 i부터 1까지 1씩 감소하며 반복하는 문법
        if array[j] < array[j-1]:
            arrya[j], array[j -1] = array[j -1], array[j] # swap
        else:
            break # 자기보다 작은 데이터를 만나면 멈춤

print(array)