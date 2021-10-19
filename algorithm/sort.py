#선택정렬
'''
처리되지 않은 데이터 중에서 가장 작은 데이터를 섲ㄴ택해 맨앞에 있는 데이터와 바꾸는 것을 반복함
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