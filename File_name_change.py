import os 

def change_filename(path, char, loc='front'):
    file_names = os.listdir(path)
    for name in file_names:
        src = os.path.join(target_path, name)
        if loc == 'front':
            dst = os.path.join(target_path, char + name)
        elif loc == 'end':
            dst = os.path.join(target_path, char + name)
        # dst = os.path.join(target_path, name.replace('4-','5-'))
        os.rename(src, dst)


# origin_path = os.getcwd()
# os.chdir(origin_path)
# target_path = r'D:\Python\강의) [FastCampus] 딥러닝 올인원 패키지\강의자료\강의자료 전체\임시폴더'
# os.listdir(target_path)



# change_filename(path=target_path, char='6-', loc='front')