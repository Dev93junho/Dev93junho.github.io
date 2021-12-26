n = int(input())
count = 0

# 큰 단위 화폐부터 차례대로 확인
array = [500, 100, 50, 10]

for coin in array:
    coin += n // coin # 해당 화폐로 거슬러 줄수 있는 동전의 걔수 세기
    n %= coin

print(count)
