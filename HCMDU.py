import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score, precision_recall_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
import math
import warnings
warnings.filterwarnings('ignore')
import struct
import utils.Evaluation_indexs as EI
from Comparison import SDUS1
from hilbertcurve.hilbertcurve import HilbertCurve
import time


def float_to_bin(number):
    """ 将浮点数转换为32位二进制字符串（基于IEEE 754标准） """
    if number == 0.0:
        return "0" * 32
    packed = np.float32(number).tobytes()
    unpacked = np.frombuffer(packed, dtype=np.uint32)[0]
    return format(unpacked, '032b')


def bin_to_float(binary_str):
    """ 将32位二进制字符串转换为浮点数（基于IEEE 754标准） """
    # 检查输入的二进制字符串长度是否为32位
    if len(binary_str) != 32:
        raise ValueError("二进制字符串必须为32位")

    # 将二进制字符串转换为32位无符号整数
    int_rep = int(binary_str, 2)

    # 将整数打包为4字节并解包为浮点数
    float_number = struct.unpack('>f', struct.pack('>I', int_rep))[0]

    return float_number


def convert_binary_series(binary_series):
    """ 将一串长的二进制数据按32位拆分并转换为浮点数 """
    # 按每32位分割二进制字符串
    floats = []
    for i in range(0, len(binary_series), 32):
        binary_chunk = binary_series[i:i + 32]
        # 检查是否刚好32位
        if len(binary_chunk) == 32:
            floats.append(bin_to_float(binary_chunk))

    return floats


def Combine_feature(X_train):
    '''二进制特征值组合'''
    # 转换二进制
    X_train = pd.DataFrame(X_train)
    X_train_bin = X_train.applymap(float_to_bin)
    # print(X_train_bin)
    # 样本二进制特征值组合
    X_train_bin_com = X_train_bin.apply(lambda row: ''.join(row), axis=1)
    # print(X_train_bin_com.apply(len))

    return X_train_bin_com


def trans(Xs, k):
    """
    将输入的样本数据 X 转换为适应希尔伯特曲线计算的格式。

    参数
    ----------
    X : array-like, shape (n, d), 输入的样本数据
    k : int, 希尔伯特曲线的阶数

    返回值
    ----------
    Xs : array-like, shape (n, d), 转换后的样本数据
    """
    # # 对输入数据 X 进行归一化处理，先按列（特征）进行最小-最大归一化
    # Xs = (X - np.min(X, 0)) / (np.max(X, 0) - np.min(X, 0))
    # 将归一化后的数据映射到 [0, 2^k] 区间
    Xs = Xs * (2 ** k)
    # 将映射后的数据转换为整数类型（浮点数转换为整数）
    Xs = np.int64(Xs)
    # 将映射后的数据中等于 2^k 的值替换为 2^k - 1
    Xs[Xs == (2 ** k)] = 2 ** k - 1

    return Xs  # 返回转换后的数据


def HCP(X, k):
    """
    实现 Hilbert 曲线投影距离（p=2），用于等样本大小和等权重情况下的计算
    参数
        X : array-like, shape (n, d), 样本在源域中的数据
        k : int, 当使用 'Hk' 时，Hilbert 曲线的阶数
    返回值
        希尔伯特曲线投影距离排序
    """
    X = np.array(X)
    n, d = X.shape  # 获取输入数据 X 和 Y 的维度，n 是样本数量，d 是每个样本的特征维度

    # 使用希尔伯特索引的方式来处理希尔伯特曲线
    p = k  # 设置希尔伯特曲线的阶数
    hilbert_curve = HilbertCurve(p, d)  # 创建一个希尔伯特曲线对象，p 是阶数，d 是维度

    # 将实例进行转换，得到适应希尔伯特曲线的形式
    Xi = trans(X, k)  # 将样本X转换为适合希尔伯特曲线计算的形式
    # print('Xi', Xi)
    # 计算样本的希尔伯特曲线距离
    Xdistances = np.array(hilbert_curve.distances_from_points(Xi)) / (2**(d*p) - 1)  # 计算每个点与希尔伯特曲线的距离，并进行归一化

    # print('距离', Xdistances)
    # Xr = np.argsort(Xdistances)  # 对样本的希尔伯特曲线距离进行排序，得到排序后的索引
    # print('排序', Xr)

    Xdistances_defect = Xdistances[-1] # 取出缺陷实例距离
    Xdistances = Xdistances[0: -1] # 取出无缺陷实例距离
    Xr = np.argsort((Xdistances - Xdistances_defect)) # 根据无缺陷实例和缺陷实例的希尔伯特距离排序
    # print('Xr', Xr)

    return Xr


def Hamming_hilbert(X_train, y_train):

    number = 0 # 多数类样本数量变化计数
    max_iter = 50 # 最大迭代次数
    num_start_noninstance = len(y_train[y_train == 0])  # 记录多数类样本数量

    X_train_min = X_train[y_train == 1]  # 少数类样本
    y_train_min = np.ones(len(X_train_min))
    num_major, num_minor = X_train[y_train == 0].shape[0], X_train[y_train == 1].shape[0]
    # print('多数类数量%d, 少数类数量%d' % (len(X_train[y_train == 0]), len(X_train[y_train == 1])))
    while len(X_train[y_train == 0]) > len(X_train[y_train == 1]) * 2:

        delete_idx = []  # 记录Hamming距离相同无缺陷实例序号
        num_delete = 0

        # 1 将实例映射到Hamming距离空间
        # 组合二进制特征值
        X_train_bin = Combine_feature(X_train)
        # 找到所有缺陷实例的索引
        X_train_min_index = np.where(y_train == 1)[0]
        # print('X_train_min_index1', X_train_min_index)
        # print('多数类数量', len(X_train[y_train == 0]))

        # same_hamming = []  # 记录在r1为半径的区域内，无缺陷实例和缺陷实例Hamming距离相同的实例数量
        random_index = random.choice(X_train_min_index)  # 随机选择一个缺陷实例(index)
        random_bin = X_train_bin[random_index]  # 二进制实例
        # print(random_index, random_bin)

        # 2-1 计算随机缺陷实例与其它缺陷实例的Hamming距离
        hamming_dis = []
        for idx in X_train_min_index:
            if idx != random_index:
                dis = sum(b1 != b2 for b1, b2 in zip(X_train_bin[random_index], X_train_bin[idx]))
                hamming_dis.append((idx, dis))
        # print(hamming_dis)
        # 找到与随机选择缺陷实例最接近的缺陷实例
        xk_index, min_dis = min(hamming_dis, key=lambda x: x[1])
        xk_bin = X_train_bin[xk_index]  # 最接近的缺陷实例
        # 计算r1
        r1 = min_dis
        # print(r1)

        # 2-2 普遍情况：在圆内寻找与随机选择缺陷实例距离最远的无缺陷实例xl
        X_train_max_index = np.where(y_train == 0)[0]  # 无缺陷实例索引
        max_dis = -1  # 内接圆半径
        xl_index = None
        found_nondefect = False  # 用于标识是否找到了无缺陷实例

        for idx in X_train_max_index:
            # 计算random_noninstance_index和xl之间的Hamming距离
            dis = sum(b1 != b2 for b1, b2 in zip(X_train_bin[random_index], X_train_bin[idx]))
            if dis <= r1:
                if dis == r1:
                    delete_idx.append(idx)  # 记录Hamming距离相同无缺陷实例序号，之后要先删除
                    # same_hamming.append(idx)
                else:
                    if dis > max_dis:
                        max_dis = dis
                        xl_index = idx
                        found_nondefect = True  # 解决r1圆内没有无缺陷实例的情况

                    # 情况3：删除与随机选择实例Hamming距离相同的无缺陷实例
                    # print('Hamming距离相同的无缺陷实例序号', delete_idx)
                    X_train_bin = X_train_bin.reset_index(drop=True)
                    delete_idx = [idx for idx in delete_idx if idx < len(X_train_bin)]  # 确保删除的索引小于数据集的长度

        if delete_idx:
            X_train_bin = X_train_bin.drop(delete_idx, axis=0).reset_index(drop=True)  # 删除这些实例的二进制特征，并重置索引
            X_train_new = pd.DataFrame(X_train).drop(delete_idx, axis=0).reset_index(drop=True)
            y_train_new = pd.DataFrame(y_train).drop(delete_idx, axis=0).reset_index(drop=True)
            X_train_new, y_train_new = np.array(X_train_new), np.array(y_train_new)
            X_train_min_index = np.where(y_train_new == 1)[0]  # 更新缺陷实例索引
            X_train_max_index = np.where(y_train_new == 0)[0]  # 更新无缺陷实例索引
            # print('X_train_new', len(X_train_new))
            # X_train_new = np.delete(X_train, delete_idx, axis=0)  # 删除这些实例的原始特征数据
            # y_train_new = np.delete(y_train, delete_idx)  # 删除这些实例的标签
            # print(len(y_train[y_train == 0]))
            # print('X_train_bin', len(X_train_bin), len(X_train_new))
            # print('X_train_min_index2', X_train_min_index)
        else:
            # print('删除前的多数类数量', len(y_train[y_train == 0]))
            # 删除距离相同，防止圆内没有无缺陷实例时报错
            X_train_bin = X_train_bin.drop(delete_idx, axis=0).reset_index(drop=True)  # 删除这些实例的二进制特征，并重置索引
            X_train_new = pd.DataFrame(X_train).drop(delete_idx, axis=0).reset_index(drop=True)
            y_train_new = pd.DataFrame(y_train).drop(delete_idx, axis=0).reset_index(drop=True)  # 删除这些实例的标签
            X_train_new, y_train_new = np.array(X_train_new), np.array(y_train_new)
            X_train_min_index = np.where(y_train_new == 1)[0]  # 更新缺陷实例索引
            X_train_max_index = np.where(y_train_new == 0)[0]  # 更新无缺陷实例索引
            # print(len(X_train_bin), len(X_train_new))
        # print('Hamming距离相同的无缺陷实例序号', delete_idx)

        # 情况2：如果圆内没有无缺陷实例，重新选择随机实例
        if not found_nondefect:
            # 如果max_iter没更新提前终止条件
            number += 1
            if number >= max_iter:
                break
            continue # 如果圆内没有无缺陷实例，重新选择随机实例
        number = 0 # 更新

        # print(xl_index)
        # 如果在r1半径内没有找到无缺陷实例，则将 max_dis设置为r1
        if xl_index is None:
            # max_dis = r1
            # xl_index = -1 # 需要设置 xl_index 为空值
            continue
        # xl_bin = X_train_bin[xl_index] # 圆内最远的缺陷实例
        # xl_bin = X_train_bin[xl_index] if xl_index != -1 else None  # 如果找不到无缺陷实例，则 xl_bin 为 None

        # 计算r2
        r2 = max_dis
        # print(r1, r2)

        # 2-3 找到圆内所有无缺陷实例
        r2_instance = []  # r2圆内所有无缺陷实例
        # print('X_train_max_index', X_train_max_index)
        if all(random_index < x for x in delete_idx):  # 判断random_index是否小于delete_idx中所有元素
            for idx in X_train_max_index:
                # 此处delete_idx对应的序号已经开始删除无缺陷实例，所以X_train_bin[random_index]发生变化。random_index可能溢出
                # print(random_index, len(delete_idx), (random_index - len(delete_idx)))
                dis = sum(b1 != b2 for b1, b2 in zip(X_train_bin[random_index], X_train_bin[idx]))
                if dis <= r2:
                    r2_instance.append(idx)  # 所有在圆内的无缺陷实例
                random_index_new = random_index # 记录random_index
        else:  # 如果delete_idx中存在序号小于random_index的元素，则random_index的序号会改变
            count = sum(1 for x in delete_idx if x < random_index)
            for idx in X_train_max_index:
                # 此处delete_idx对应的序号已经开始删除无缺陷实例，所以X_train_bin[random_index]发生变化。random_index可能溢出
                # print(random_index, len(delete_idx), (random_index - len(delete_idx)))
                dis = sum(b1 != b2 for b1, b2 in zip(X_train_bin[random_index - count], X_train_bin[idx]))
                if dis <= r2:
                    r2_instance.append(idx)  # 所有在圆内的无缺陷实例
                random_index_new = random_index - count
        # print(r1, r2, delete_idx, r2_instance)

        # 情况2：如果圆内没有无缺陷实例，重新选择随机实例
        if len(r2_instance) == 0:
            continue

        # 计算区域内所需删除无缺陷实例数量
        circle_num_major = len(r2_instance)
        ratio = circle_num_major / num_major
        num_delete = math.ceil(num_minor * ratio)
        # print('区域内删除数量', num_delete)

        # 将随机选择的缺陷实例加入r2_instance中
        r2_instance_defect = r2_instance
        r2_instance_defect.append(random_index_new) # random_index需要处理，缺陷实例在最后一个
        # print('圆内所有实例', r2_instance_defect)
        # 圆内实例转换为浮点数
        binary_string = X_train_bin.iloc[r2_instance_defect].values
        # print(binary_string)
        X_train_float = np.array([convert_binary_series(bin_str) for bin_str in binary_string])
        # print(X_train_float)

        # # 圆内实例转换为浮点数
        # binary_string = X_train_bin.iloc[r2_instance].values
        # # print(binary_string)
        # X_train_float = np.array([convert_binary_series(bin_str) for bin_str in binary_string])
        # # print(X_train_float)

        # 3 计算希尔伯特曲线投影距离。探索圆内多数类样本的分布规律
        select_index = HCP(X_train_float, k=5)
        # 记录需要删除样本的序号
        sort_index = sorted(range(len(select_index)),
                            key=lambda i: select_index[i])  # 返回r2_instance按顺序排列后，select_index[i]对应i的顺序
        # print(sort_index)
        # sort_index_delete = [r2_instance[i] for i in sort_index[: (num_delete - len(delete_idx))]] # 删除样本的序号，还有删除相同Hamming距离的无缺陷实例
        sort_index_delete = [r2_instance[i] for i in sort_index[: (num_delete)]]  # 删除样本的序号，还有删除相同Hamming距离的无缺陷实例
        # print('删除序号', sort_index_delete)

        # 删除与随机选择实例Hamming距离相同的无缺陷实例
        X_train_bin = X_train_bin.drop(sort_index_delete, axis=0).reset_index(drop=True)
        X_train_new = pd.DataFrame(X_train_new).drop(sort_index_delete, axis=0).reset_index(drop=True)
        y_train_new = pd.DataFrame(y_train_new).drop(sort_index_delete, axis=0).reset_index(drop=True)
        X_train_new, y_train_new = np.array(X_train_new), np.array(y_train_new)
        y_train_new = y_train_new.flatten()  # 转换为一维数组，或者使用.ravel()
        # print('删除后', len(y_train_new[y_train_new == 0]), len(X_train_bin[y_train_new == 0]))
        # print('删除后多数类数量', len(y_train_new[y_train_new == 0]))

        X_train = X_train_new
        y_train = y_train_new
        # print(len(y_train[y_train == 0]), len(y_train[y_train == 1]))

    # 删除后的多数类样本
    X_train_maj = X_train[y_train == 0]
    y_train_maj = np.zeros(len(X_train_maj))
    X_train_con = np.concatenate((X_train_min, X_train_maj), axis=0)
    y_train_con = np.concatenate((y_train_min, y_train_maj), axis=0)

    return X_train_con, y_train_con


if __name__ == '__main__':

    start_time = time.time()

    data_frame = np.array(pd.read_csv(r'D:\A论文实验\data\JIRA\hive-0.9.0.csv'))
    print('样本个数: ', data_frame.shape[0], '特征个数: ', data_frame.shape[1] - 1)
    data = data_frame[:, : -1]
    target = data_frame[:, -1]
    print('缺陷率：', np.sum(target == 1) / data.shape[0])

    # 归一化
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    n_split = 5
    kfold = KFold(n_splits=n_split, shuffle=True, random_state=42)

    F_measure, TPR, FPR, FNR, G_mean, MCC, AUC = [], [], [], [], [], [], []
    F_measure_smo, FPR_smo, FNR_smo, G_mean_smo, MCC_smo, AUC_smo = [], [], [], [], [], []
    for kf, (train_index, test_index) in enumerate(kfold.split(data)):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = target[train_index], target[test_index]
        # print('多数类数量%d, 少数类数量%d' % (len(X_train[y_train == 0]), len(X_train[y_train == 1])))

        X_train_con, y_train_con = Hamming_hilbert(X_train, y_train)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train_con, y_train_con)
        pred = model.predict(X_test)
        tpr, fpr, fnr, gmean, mcc, auc = EI.evaluation_indexs(y_test, pred)
        TPR.append(tpr)
        FPR.append(fpr)
        FNR.append(fnr)
        G_mean.append(gmean)
        MCC.append(mcc)
        AUC.append(auc)
        print('*' * 50)

        # model =

    print(G_mean, MCC, AUC, FPR, FNR)
    print('*' * 25, 'RF', '*' * 25)
    print('希尔伯特 G-mean: %.4f' % np.mean(G_mean))
    print('希尔伯特 MCC: %.4f' % np.mean(MCC))
    print('希尔伯特 AUC: %.4f' % np.mean(AUC))
    print('希尔伯特 FPR: %.4f' % np.mean(FPR))
    print('希尔伯特 FNR: %.4f' % np.mean(FNR))

    end_time = time.time()
    print('总时间%ds' % (end_time - start_time))









