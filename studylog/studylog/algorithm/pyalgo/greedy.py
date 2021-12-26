#-*-coding: utf-8 -*-
'''
1. 문제의 최적 부분 구조 결정
2. 재귀해를 만든다
3. 그리디 선택을 하고나면 한개의 부분문제만 남음
4. 그리디 방법론을 구현하는 재귀 구조를 만듦
5. 재귀 구조가 반복되는 알고리즘 
'''

# 최소 동전 거슬러주기

def min_coin_count(value, coin_list):
    count = 0
    changed_list = sorted(coin_list, reverse=True)
    for i in changed_list:
        count += (value//i)
        value = value % i
    return count


# test case
default_coin_list = [100, 500, 10, 50]
print(min_coin_count(1440, default_coin_list))
print(min_coin_count(1700, default_coin_list))
print(min_coin_count(23520, default_coin_list))
print(min_coin_count(32590, default_coin_list))