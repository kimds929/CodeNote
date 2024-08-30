import datetime

# 날짜 생성
datetime.datetime(2021, 9, 21, 12, 30 ,45)  # 2021-09-21 12:30:45


# 현재날짜
datetime.datetime.today()
datetime.datetime.now()

print(datetime.datetime.now())  # 2020-01-07 15:40:15.087337 

now = datetime.datetime.now()
print(now)              # 2020-01-07 15:40:15.087337


# 특정 년/월/일
now = datetime.datetime.now()
print(now.year)         # 2020
print(now.month)        # 1
print(now.day)          # 7
print(now.hour)         # 15
print(now.minute)       # 40
print(now.second)	    # 15
print(now.microsecond)  # 087337


# 날짜이동
now = datetime.datetime.now()
print(now + datetime.timedelta(weeks=1))        # 2020-01-14 15:40:15.087337
print(now + datetime.timedelta(days=1))         # 2020-01-08 15:40:15.087337
print(now + datetime.timedelta(hours=5, minutes=10))    # 2020-01-07 20:50:15.087337


# 지난달
from dateutil.relativedelta import relativedelta

now = datetime.datetime.now()
last_month = now - relativedelta(months=1)
last_month = last_month.strftime('%Y/%m')

# format transformation
now = datetime.datetime.now()

# date → string
print(now.strftime('%Y-%m-%d'))             # 2020-01-07
print(now.strftime('%H:%M:%S'))             # 15:40:15
print(now.strftime('%Y-%m-%d %H:%M:%S'))    # 2020-01-07 15:40:15

# string → date
now = datetime.datetime.now()
print(datetime.datetime.strptime('2020-01-07 15:40:15', '%Y-%m-%d %H:%M:%S'))   # 2020-01-07 15:40:15
print(type(datetime.datetime.strptime('2020-01-07 15:40:15', '%Y-%m-%d %H:%M:%S')))# <class 'datetime.datetime'>

# date → numeric(day)
base_dt = datetime.datetime(1900, 1, 1)
now = datetime.datetime.now()
time_delta = (now - base_dt)
time_delta.days + time_delta.seconds/(24*60*60)



# 요일
def what_day_is_today(self):
    now = datetime.datetime.now()
    t = ['월', '화', '수', '목', '금', '토', '일']
    r = datetime.datetime.today().weekday()
    day = str(now.year) + '년 ' + str(now.month) + '월 ' + str(now.day) + '일 ' + t[r] + '요일'
    return day
