from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from filterpy.kalman import KalmanFilter

# 生成1000个位置,从1到1000是小车的实际位置
z = np.linspace(1,1000,1000)
# 添加噪声
mu,sigma = 0 ,1
noise = np.random.normal(mu,sigma,1000)
# 小车位置的观测值
z_nosie = z+noise


# 卡尔曼滤波器的参数初始化
my_filter = KalmanFilter(dim_x=2,dim_z=1)
#定义卡尔曼滤波中所需的参数
# x 初始状态为(0,0),即初始位置为0,速度为0
# 这个初始值不是非常重要，在利用观测值进行更新迭代后会接近真实值
my_filter.x = np.array([[0.],[0.]])

# P 协方差矩阵,表示状态向量内位置与速度的相关性
# 假设速度和位置没关系，协方差矩阵为[[1,0],[0,1]]
my_filter.P = np.array([[1.0,0.],[0.,1.]])

# F初始的状态转移矩阵，假设为匀速运动模型，可将其设置为如下:
my_filter.F = np.array([[1.,1.],[0.,1.]])

# Q 为状态转移协方差矩阵，也就是外界噪声
# 在该例中假设小车匀速，外界干扰小，所以我们对F非常确定，觉得F一定不会出错,所以Q设的很小
my_filter.Q = np.array([[0.0001,0.],[0.,0.0001]])

# 观测矩阵 Hx = P
# 利用观测数据对预测进行更新，观测矩阵的左边一项不能设置成0,量纲转化误差
my_filter.H = np.array([[1,0]])

# R 测量噪声,方差为1
my_filter.R = 1

z_new_list = []
v_new_list = []

for k in range(len(z_nosie)):
    # 预测过程
    my_filter.predict()
    # 利用观测值进行更新
    my_filter.update(z_nosie[k])
    # do something with the output
    x = my_filter.x
    # 收集卡尔曼滤波后的速度和位置信息
    z_new_list.append(x[0][0])
    v_new_list.append(x[1][0])


# 位移的偏差
dif_list = []
for k in range(len(z)):
    dif_list.append(z_new_list[k]-z[k])
# 速度的偏差
v_dif_list = []
for k in range(len(z)):
    v_dif_list.append(v_new_list[k]-1)

plt.figure(figsize=(20,9))
plt.subplot(1,2,1)
plt.xlim(-50,1050)
plt.ylim(-3.0,3.0)
plt.scatter(range(len(z)),dif_list,color='blue',label='位置偏差')
plt.scatter(range(len(z)),v_dif_list,color='yellow',label='速度偏差')
plt.legend()
plt.show()


