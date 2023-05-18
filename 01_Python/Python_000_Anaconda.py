# < 아나콘다 버전 업데이트 >
# conda update conda
# conda update anacond


# < 아나콘다 가상환경 phthon 버전 변경 >
# 0. source activate (가상환경명) : 가상환경 실행.
# 1. python -V : 파이썬 버전 확인.
# 2. conda search python : 사용 가능한 python list 확인.
# 3. conda install python=x.x.x : 입력 버전으로 파이썬 버전이 변경됨.
# 4. source deactivate
# 5. source activate (가상환경명) : 가상환경 실행.
# 6. python -V : 변경된 파이썬 버전을 확인할 수 있음.


# < 가상환경 이름변경 : 복사 → 삭제 >
# Anaconda를 사용하면서 항상 가상환경을 활성화 시켜놓고 프로그래밍을 하는 편인데, 처음에 이름을 너무 복잡하게 해서 이름을 변경하기 위해 검색해보니, 가상환경의 이름을 변경하는 기능은 없다고 한다.
# 그래서, 현재 사용하는 가상환경을 다른 이름으로 복사하고, 그 전에 사용하던 가상환경은 삭제하는 식의 방법을 사용할 수 있다.

# conda create --name [변경할 이름] --clone [기존 환경 이름]
# conda activate [변경할 이름]
# conda remove --name [기존 환경 이름] --all
