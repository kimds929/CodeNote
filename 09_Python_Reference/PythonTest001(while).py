
coffee = 10
while True:
    money = int(input("돈을넣어주세요: "))
    if money == 300:
        print("커피를줍니다.")
        coffee = coffee-1
        print("남은 커피의양은 %d개입니다." % coffee)
    elif money >300:
        print("거스름돈 %d를 주고 커피를 줍니다." %(money-300))
        coffee = coffee-1
        print("남은 커피의양은 %d개입니다." % coffee)
    elif money == 0:
        print("자판기게임을 종료합니다.")
        break
    else:
        print("돈이 모자라서 돈을 돌려주고 커피를 주지 않습니다.")
        print("남은 커피의양은 %d개입니다." %coffee)
    if not coffee:
        print("커피가 다떨어졌습니다. 판매를 중지합니다.")
        break