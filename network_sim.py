import random
import matplotlib.pyplot as plt
import numpy as np
import scipy
import json
import copy
import argparse
import math
import matplotlib
# 获取基站位置
def hexagon_centers(K, Cell_layout):
    ISD = Cell_layout['ISD']
    centers = np.zeros((K, 3))
    centers[:, 2] = Cell_layout['H_BS']
    for i in range(K):
        if i == 0:
            centers[i, 0:2] = 0
        if 0 < i <= 6:
            centers[i, 0] = ISD * math.cos(i * math.pi/3 - math.pi/6)
            centers[i, 1] = ISD * math.sin(i * math.pi/3 - math.pi/6)
        if 6 < i <= 18:
            if i % 2 == 1:
                centers[i, 0] = 2 * ISD * math.cos((i - 5)/2 * (math.pi/3) - math.pi/6)
                centers[i, 1] = 2 * ISD * math.sin((i - 5)/2 * (math.pi/3) - math.pi/6)
            if i % 2 == 0:
                centers[i, 0] = 2 * ISD * math.cos(math.pi/6) * math.cos((i-6)/2 * math.pi/3)
                centers[i, 1] = 2 * ISD * math.cos(math.pi/6) * math.sin((i-6)/2 * math.pi/3)
    return centers
# 获取网络用户信息
def get_UE(BS_loca, cell_layout):
    #随机分布 极坐标系
    N = cell_layout['N']
    K = cell_layout['K']
    radius = cell_layout['ISD']/2 * math.cos(math.pi/6)
    num = int(N/K)
    UE_loca = np.zeros((N, 3))
    UE_loca_ji = np.random.rand(N, 2)
    UE_loca_ji[:, 0] = UE_loca_ji[:, 0]*0.9 + 0.1
    for i in range(N):
        UE_loca[i, 0] = BS_loca[i % K, 0] + UE_loca_ji[i, 0] * radius * math.cos(UE_loca_ji[i, 1] * 2 * math.pi)
        UE_loca[i, 1] = BS_loca[i % K, 1] + UE_loca_ji[i, 0] * radius * math.sin(UE_loca_ji[i, 1] * 2 * math.pi)
        if random.random() < cell_layout['Indoor_UT']:
            UE_loca[i, 2] = np.random.uniform(1,np.random.uniform(4, 8))
        else:
            UE_loca[i, 2] = 1
    return UE_loca
# 路径损耗 大尺度衰落
def get_path_loss(celluar_net,Cell_layout):
    f_c = 2000000000.0
    N = Cell_layout['N']
    K = Cell_layout['K']
    BS_loca = celluar_net['BS_loca']
    UE_loca = celluar_net['UE_loca']
    LSF = np.zeros((N, N))
    d_BP = 35
    for i in range(N):
        for j in range(N):
            d_2D = np.linalg.norm(BS_loca[i % K, 0:2] - UE_loca[j, 0:2])
            d_3D = np.linalg.norm(BS_loca[i % K, :] - UE_loca[j, :])
            if d_3D < d_BP:
                LSF[i, j] = - (28.0 + 22 * math.log10(d_3D) + 20 * math.log10(f_c))
            else:
                LSF[i, j] = -(28.0 + 40 * math.log10(d_3D) + 20 * math.log10(f_c) - 9 * math.log10(
                    d_BP**2 + (BS_loca[i % K, 2] - UE_loca[j, 2])**2))
    return LSF

def main(args):

    scenarios_file = args.scenarios_file

    # 网络配置参数
    with open('./config/scenarios/' + scenarios_file + '.json', 'r') as f:
        scenarios_para = json.load(f)
    Cell_layout = scenarios_para['Cell_layout']
    # 储存网络信息
    celluar_net = {}
    # 蜂窝网初始结构
    celluar_net['BS_loca'] = hexagon_centers(Cell_layout['K'], Cell_layout)
    celluar_net['UE_loca']= get_UE(celluar_net['BS_loca'], Cell_layout)
    LSF = get_path_loss(celluar_net=celluar_net, Cell_layout=Cell_layout)
    # 可视化
    fig = plt.figure('networks')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(celluar_net['BS_loca'][:, 0], celluar_net['BS_loca'][:, 1], celluar_net['BS_loca'][:, 2], s=40)
    index_in = np.where(celluar_net['UE_loca'][:, 2] > 1)[0]
    index_out = np.where(celluar_net['UE_loca'][:, 2] <= 1)[0]
    ax.scatter(celluar_net['UE_loca'][index_in, 0], celluar_net['UE_loca'][index_in, 1], celluar_net['UE_loca'][index_in, 2], s=15)
    ax.scatter(celluar_net['UE_loca'][index_out, 0], celluar_net['UE_loca'][index_out, 1], celluar_net['UE_loca'][index_out, 2], s=15)
    ax.set_box_aspect([10, 10, 2])
    plt.show()
if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='give test scenarios.')

    parser.add_argument('--scenarios-file', type=str, default='UMa-19',
                        help='scenarios file for the deployment')
    args = parser.parse_args()
    main(args)
