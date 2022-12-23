
# [matplotlib 사용법] ------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# matplotlib 한글폰트사용
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)



# plt.figure(figsize=(10,6)) 
# plt.plot(x, y)
# plt.show()

# https://matplotlib.org/api/pyplot_api.html        # 참고 공식 사이트
# https://datascienceschool.net/view-notebook/d0b1637803754bb083b5722c9f2209d0/     # 참고 Site
# https://wikidocs.net/book/5011    # Matplotlib Tutorial - 파이썬으로 데이터 시각화하기

# color, edgecolors
    # blue	b
    # green	g
    # red	r
    # cyan	c
    # magenta	m
    # yellow	y
    # black	k
    # white	w

# marker
    # .	point marker
    # ,	pixel marker
    # o	circle marker
    # v	triangle_down marker
    # ^	triangle_up marker
    # <	triangle_left marker
    # >	triangle_right marker
    # 1	tri_down marker
    # 2	tri_up marker
    # 3	tri_left marker
    # 4	tri_right marker
    # s	square marker
    # p	pentagon marker
    # *	star marker
    # h	hexagon1 marker
    # H	hexagon2 marker
    # +	plus marker
    # x	x marker
    # D	diamond marker
    # d	thin_diamond marker

# linestyle
    # -	solid line style
    # --	dashed line style
    # -.	dash-dot line style
    # :	dotted line style

# style
    # color(c)	선 색깔
    # linewidth(lw)	선 굵기
    # linestyle(ls)	선 스타일
    # marker		마커 종류
    # markersize(ms)	마커 크기
    # markeredgecolor(mec)	마커 선 색깔
    # markeredgewidth(mew)	마커 선 굵기
    # markerfacecolor(mfc)	마커 내부 색깔

# subplot

# fig.subplots_adjust(hspace=0.8)   # subplot간격
# left = 0.125  # the left side of the subplots of the figure
# right = 0.9   # the right side of the subplots of the figure
# bottom = 0.1  # the bottom of the subplots of the figure
# top = 0.9     # the top of the subplots of the figure
# wspace = 0.2  # the amount of width reserved for space between subplots,
#               # expressed as a fraction of the average axis width
# hspace = 0.2  # the amount of height reserved for space between subplots,
#               # expressed as a fraction of the average axis height

# sub_line

# annotate


plt.figure(figsize=(10,6)) 
plt.plot([10, 20, 30, 40], [1, 4, 9, 16], c="b",
         lw=5, ls="--", marker="o", ms=15, mec="g", mew=5, mfc="r", drawstyle='steps-post')
plt.title("스타일적용예")
plt.show()


    # list 값을 이용한 line Graph
plt.figure(figsize=(10,6)) 
plt.plot([1,2,3,4,5,6,7,9,10,9,7,6,5,4,3,2,1,0])
plt.show()

    # sin함수 곡선 Graph + grid + title/label ------------------------------------------------------------------
plt.figure(figsize=(10,6)) 
t = np.arange(0,12,0.1)
y = np.sin(t)
plt.plot(t, y)
plt.grid() # 그리드 적용하기
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.title('Example of sinewave')
plt.show()

    # sin함수 + cosine함수 곡선 Graph ------------------------------------------------------------------
plt.figure(figsize=(10,6))
plt.plot(t, np.sin(t))
plt.plot(t, np.cos(t))
plt.grid()
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.title('Example of sinewave')
plt.show()


    # Line 색상, 굵기 +  Legend ------------------------------------------------------------------
plt.figure(figsize=(10,6))
plt.plot(t, np.sin(t), label='sin', lw=3, marker='o')   # lw : lineWidth
plt.plot(t, np.cos(t), color='coral', label='cos', linestyle='dashed')  # color : lineColor
plt.grid()
plt.legend()
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.title('Example of sinewave')
plt.ylim(-2, 2)     # y range : -2~2
plt.xlim(0, 2*np.pi)      # x range : 0~2π
plt.show()

    # plt.scatter / colorbar ------------------------------------------------------------------
t = np.arange(0, 5, 0.5)
plt.figure(figsize=(10,6))
plt.plot(t, t, 'ro')        # red, o
plt.plot(t, t**2, 'bs-')    # blue, square, line
plt.plot(t, t**3, 'g^--')   # green, triangle, dashed-line
plt.plot(t, 10*t, 'y*:')      # yellow, star, dotted line 
plt.scatter(t, 50*np.sqrt(t), s=50, c=t, marker='p')
plt.colorbar()
plt.show()


    # line-plot / Box-plot ------------------------------------------------------------------
s1 = np.random.normal(loc=0, scale=1, size=1000)
s2 = np.random.normal(loc=5, scale=0.5, size=1000)
s3 = np.random.normal(loc=10, scale=2, size=1000)


    # line-plot ------------------------------------------------------------------
plt.figure(figsize=(10,6))
plt.plot(s1, label='s1')
plt.plot(s2, label='s2')
plt.plot(s3, label='s3')
plt.legend()
plt.show()

    # box-plot ------------------------------------------------------------------
plt.figure(figsize=(10,6))
plt.boxplot((s1, s2, s3))
plt.grid()
plt.show()




# Color Setting ------------------------------------------------------------------
import matplotlib as mpl
# mpl.rcParams['image.cmap'] = 'viridis'  # (default) Blue → Green → Yellow
# mpl.rcParams['image.cmap'] = 'jet'      # Rainbow:  Blue → Green → Yellow → Red → Brown
# mpl.rcParams['image.cmap'] = 'RdBu'     # Red → Red
# mpl.rcParams['image.cmap'] = 'coolwarm' # Blue → Red
plt.scatter(s1, s2, c=s3)
plt.colorbar()








    # 행렬 Matrix 구현 ------------------------------------------------------------------
a_matrix = np.random.rand(5,5)

fig_matrix, axs_matrix = plt.subplots(1, 3, figsize=(10,3))

for ax, interp in zip(axs_matrix, ['nearest', 'bilinear', 'bicubic']):
        # nearest : block단위
        # bilinear : linear형태로 Gradient하게
        # bicubic : 2차식 형태로 Gradient하게
    ax.imshow(a_matrix, interpolation=interp)
    ax.set_title(interp.capitalize())
    ax.grid(True)
plt.show()

# 'antialiased', 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos', 'blackman'


    # 3차원 상의 표면그리기 ------------------------------------------------------------------
from mpl_toolkits.mplot3d import Axes3D     # 실제로 사용하고 있지 않지만 import되어야 3d map이 생성됨
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

fig_3d = plt.figure()
ax_3d = fig_3d.gca(projection='3d')

# Make Data
x_3d = np.arange(-5, 5, 0.25)
y_3d = np.arange(-5, 5, 0.25)
x_3d_mesh, y_3d_mesh = np.meshgrid(x_3d, y_3d)      # x * y 만큼 반복을 시켜줌

r_3d = np.sqrt(x_3d_mesh**2, y_3d_mesh**2)
z_3d = np.sin(r_3d)         # 모든 x, y좌표에 대한 z값

# Plot the surface
surface_3d = ax_3d.plot_surface(x_3d_mesh, y_3d_mesh, z_3d, cmap=cm.coolwarm,
                                linewidth=0, antialiased=False)
# Customize the z axis.
ax_3d.set_zlim(-1.01, 1.01)
ax_3d.zaxis.set_major_locator(LinearLocator(10))
ax_3d.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig_3d.colorbar(surface_3d, shrink=0.5, aspect=5)

plt.show()


    # 화살표(Quiver) 표현하기 : Gradient를 표현할때 유용함 --------------------------------------------
x_quiver = np.arange(-10, 10, 1)
y_quiver = np.arange(-10, 10, 1)
x_quiver_mesh, y_quiver_mesh = np.meshgrid(x_quiver, y_quiver)      # 각 X, Y 위치에 이동벡터를 넣어줌

fig_quiver, ax_quiver = plt.subplots()
q = ax_quiver.quiver(x_quiver, y_quiver, x_quiver_mesh, y_quiver_mesh)
ax_quiver.quiverkey(q, X=0.3, Y=1.1, U=10, label='Quiver key, length=10', labelpos='E')

plt.show()






    # mesh grid --------------------------------------------
x_mesh = np.linspace(0,10,66)
y_mesh = np.linspace(10,20,66)

xs, ys = np.meshgrid(x_mesh, y_mesh)
k = np.hstack([xs.reshape(-1,1), ys.reshape(-1,1)])

cond = (k.sum(1) > 20) & (k.sum(1) < 25)
# cond = k[:,0]**2 + k[:,1]**2 < np.square(15)
# cond = (k[:,0]-5)**2 + (k[:,1]-15)**2 < np.square(3)
# cond = (np.abs(k[:,0]-5) +np.abs(k[:,1]-15)<3)

plt.figure()
plt.plot(k[:,0], k[:,1], 'k', alpha=0.1, markeredgecolor='white')
plt.plot(k[cond][:,0], k[cond][:,1], 'ro', alpha=0.1, markeredgecolor='white')
plt.show()



# 그래프 파일로 저장하기
test_data = [1,2,3,4]
fig = plt.figure()
plt.plot(test_data)
plt.show()

plt.savefig('save_name.jpg')