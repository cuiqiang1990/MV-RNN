#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:       Qiang Cui:  <cuiqiang1990[at]hotmail.com>
# Descripton:   load the dataset
# Create Date:  2016-12-05 10:00:00
# Modify Date:  2016-00-00 00:00:00
# Modify Disp:

from __future__ import print_function
import time
import numpy as np
import pandas as pd
import random
from collections import Counter
import math
from copy import deepcopy
import glob
import json
from numpy.random import binomial
__docformat__ = 'restructedtext en'


def exe_time(func):
    def new_func(*args, **args2):
        t0 = time.time()
        print("-- @%s, {%s} start" % (time.strftime("%X", time.localtime()), func.__name__))
        back = func(*args, **args2)
        print("-- @%s, {%s} end" % (time.strftime("%X", time.localtime()), func.__name__))
        print("-- @%.3fs taken for {%s}" % (time.time() - t0, func.__name__))
        return back
    return new_func


def load_data(dataset, mode, split, intervals):
    """
    加载购买记录文件，生成数据。
    """
    # 用户购买历史记录，原纪录. 嵌套列表, 元素为一个用户的购买记录(小列表)
    # 同时保留好购买时间的最大最小值，用于以此为准切分数据。
    print('Original data ...')
    buys = pd.read_csv(dataset, sep=' ')
    all_user_buys = [[str(i) for i in buy.split(',')] for buy in buys['buys']]          # 后边用
    all_user_stps = [[int(i) for i in buy.split(',')] for buy in buys['time_stamps']]   # 后边用
    all_trans = [item for buys in all_user_buys for item in buys]
    all_stmps = [stmp for stps in all_user_stps for stmp in stps]
    tran_num, user_num, item_num = len(all_trans), len(all_user_buys), len(set(all_trans))
    stmp_min = min(all_stmps)
    stmp_max = max(all_stmps)
    date_min = time.strftime("%Y.%m.%d %H:%M:%S", time.localtime(stmp_min))
    date_max = time.strftime("%Y.%m.%d %H:%M:%S", time.localtime(stmp_max))
    print('\tusers, items, trans:    = {v1}, {v2}, {v3}'.format(v1=user_num, v2=item_num, v3=tran_num))
    print('\ttime_stamp min, max:    = [{v1}, {v2}]'.format(v1=stmp_min, v2=stmp_max))
    print('\ttime_date  min, max:    = [{v1}, {v2}]'.format(v1=date_min, v2=date_max))
    print('\tavg. user buy:          = {val}'.format(val=1.0 * tran_num / user_num))
    print('\tavg. item bought:       = {val}'.format(val=1.0 * tran_num / item_num))
    # Amazon只选用最近2年的交易数据，其交易量：1311204/1561102 = 84.0%(a5)
    # stmp_max = 1406073600,                # 2014.07.23_08.00.00
    if 'amazon' in dataset:
        stmp_min = 1406073600 - 63072000    # 2012.07.23_08.00.00   # 1343001600
        date_min = time.strftime("%Y.%m.%d %H:%M:%S", time.localtime(stmp_min))
    per1 = int(stmp_min + (stmp_max - stmp_min) * split[0])     # 0.6 / 0.8
    per2 = int(stmp_min + (stmp_max - stmp_min) * split[1])     # 0.8 / 1.0

    # # 做测试：把all_stmps划分为12个区间，看看各区间各有多少交易量。据此选定'Amazon'的时间区间(近2年内)。已测试通过
    # nums = 11
    # space = (stmp_max - stmp_min) / nums
    # inters = np.array([0 for _ in range(nums)])
    # for stmp in all_stmps:
    #     if stmp == stmp_min:        # 不对这个做判断的话，那些stmp=stmp_min的，会被加到区间inters[-1]里去。
    #         inters[0] += 1
    #     elif stmp > stmp_min:
    #         inter = int(math.ceil(1.0 * (stmp - stmp_min) / space)) - 1   # 按交易时间计算落在哪个区间
    #         inters[inter] += 1
    # print(inters, sum(inters))
    # # 做测试：查看amazon最近2年的交易占总量的百分比。
    # a = 0
    # for stmp in all_stmps:
    #     if stmp >= stmp_min:
    #         a += 1
    # print(a)
    # print(1.0 * a / tran_num)
    # # 做测试：区间划分里各区间的交易量是否正确。已测试通过
    # k = 0
    # for stmp in all_stmps:
    #     if stmp > per80:        # 采用per80时，nums=5
    #         k += 1
    # print(k)                    # 测试：k = inters[-1]，正确。

    # 选取训练集、验证集或训练集、测试集，并对test去重。
    # 不管是valid还是test模式，统一用train，test表示。
    print('Remove duplicates in test set: mode = {val} ...'.format(val=mode))
    tra_buys, tes_buys = [], []
    only_left, only_right = 0, 0
    for ubuys, ustps in zip(all_user_buys, all_user_stps):
        # 按购买时间为准进行切分。不采用选取每个购买序列前边的多少
        split0 = int(len([stmp for stmp in ustps if stmp < stmp_min]))  # 序列从stmp_min对应的item开始。
        split1 = int(len([stmp for stmp in ustps if stmp <= per1]))
        split2 = int(len([stmp for stmp in ustps if stmp <= per2]))
        # valid: 60%train, 20%valid -> [0, 60], (60, 80]
        # test:  80%train, 20%test  -> [0, 80], (80, 100]
        left, right = ubuys[split0: split1], ubuys[split1: split2]
        # 丢掉切分后只有left或只有right的user。
        if not left:
            only_right += 1
            continue
        if not right:
            only_left += 1
            continue
        # test需要自身去重。test只要有元素，right_stay就不会为空。
        right_stay = list(set(right))       # 先去重
        right_stay.sort(key=right.index)    # test里剩余的按原顺序排
        # 保存
        tra_buys.append(left)
        tes_buys.append(right_stay)

    # 去重后的基本信息，
    all_trans = []
    for utra, utes in zip(tra_buys, tes_buys):
        all_trans.extend(utra)
        all_trans.extend(utes)
    tran_num, user_num, item_num = len(all_trans), len(tra_buys), len(set(all_trans))
    print('\tusers, items, trans:    = {v1}, {v2}, {v3}'.format(v1=user_num, v2=item_num, v3=tran_num))
    print('\ttime_stamp min, max:    = [{v1}, {v2}]'.format(v1=stmp_min, v2=stmp_max))
    print('\ttime_date  min, max:    = [{v1}, {v2}]'.format(v1=date_min, v2=date_max))
    print('\tavg. user buy:          = {val}'.format(val=1.0 * tran_num / user_num))
    print('\tavg. item bought:       = {val}'.format(val=1.0 * tran_num / item_num))
    print('\tuseful, only left, only right: = {v0} + {v1} + {v2} = {v4}'
          .format(v0=user_num, v1=only_left, v2=only_right, v4=user_num + only_left + only_right))

    # 先按test里各list长度排序(minor sort)，再按train里各list排序(major sort)。
    tes_sorted = sorted(zip(tra_buys, tes_buys), key=lambda x: len(x[1]), reverse=True)
    tra_sorted = sorted(tes_sorted, key=lambda x: len(x[0]), reverse=True)
    tra_buys, tes_buys = zip(*tra_sorted)

    # 建立商品别名字典。更新购买记录，替换为0~len(se)-1的别名
    print('Use aliases to represent items ...')
    all_items = set(all_trans)
    aliases_dict = dict(zip(all_items, range(item_num)))
    tra_buys = [[aliases_dict[i] for i in buy] for buy in tra_buys]
    tes_buys = [[aliases_dict[i] for i in buy] for buy in tes_buys]

    # 测试集指标的分母计算: test集中items按次数划分为区间，
    # （1）各区间有哪些items，（2）各区间累计有多少trans，（3）cold/avtive的items、trans各有多少
    print('Devide test items into intervals ...')
    tra = [item for buy in tra_buys for item in buy]
    tes = [item for buy in tes_buys for item in buy]   # 用别名表示的所有test items
    test_i_cou = dict(Counter(tes))                    # key是测试集里商品别名，value是它出现的次数
    test_i_intervals_items = [[] for _ in range(intervals[1])]  # 某区间里有哪些items，
    test_i_intervals_trans = [0 for _ in range(intervals[1])]   # 区间里的items对应有多少transactions，算评价指标用。
    for item in tes:
        inter = int(math.ceil(1.0 * test_i_cou[item] / intervals[0])) - 1   # 按被购买次数计算落在哪个区间
        if inter >= intervals[1] - 1:   # inter >= 最大的区间标号
            inter = -1
        test_i_intervals_items[inter].append(item)
        test_i_intervals_trans[inter] += 1
    test_i_intervals_items = [len(set(one_inter)) for one_inter in test_i_intervals_items]
    print('\tintervals               = {val}'.format(val=intervals))
    print('\ttest_i_intervals_items  = {v1}, {v2}'.format(v1=sum(test_i_intervals_items), v2=test_i_intervals_items))
    print('\ttest_i_intervals_trans  = {v1}, {v2}'.format(v1=sum(test_i_intervals_trans), v2=test_i_intervals_trans))
    tmp1 = [sum(test_i_intervals_items[:2]), sum(test_i_intervals_items[2:])]
    tmp2 = [sum(test_i_intervals_trans[:2]), sum(test_i_intervals_trans[2:])]   # <=4, >=5
    print('\ttest_i_cold_active_items= {v1}, {v2}'.format(v1=sum(tmp1), v2=tmp1))
    print('\ttest_i_cold_active_trans= {v1}, {v2}'.format(v1=sum(tmp2), v2=tmp2))
    assert sum(tmp1) == len(set(tes))   # test item: unique items
    assert sum(tmp2) == len(tes)        # test item: bought times
    # 下面两个array，是要根据各个interval里的交易数，来算AUC、Recall@30的区间命中率的。
    test_i_intervals_cumsum = np.asarray(test_i_intervals_trans).cumsum()
    test_i_cold_active = np.asarray(tmp2)
    assert test_i_intervals_cumsum[-1] == sum(test_i_cold_active)
    # test_i_cold_active = np.asarray([sum(test_i_intervals_trans[:2]),       # <= 4
    #                                  sum(test_i_intervals_trans[2:])])      # >= 5 明明是temp2
    set_tra = set(tra)
    set_tes = set(tes)  # 用于把tes里其中一部分item的图文特征给清零。
    # check: 用于查看tes独有多少items，与tra共有多少items。
    # print(len(set_tra), len(set_tes))
    # print(len(set_tra - set_tes), len(set_tra & set_tes), len(set_tes - set_tra))
    return [(user_num, item_num), aliases_dict,
            (test_i_cou, test_i_intervals_cumsum, test_i_cold_active),
            (tra_buys, tes_buys), (set_tra, set_tes)]


def load_img_txt(name, it_fea_path, item_num, it_size, aliases_dict):
    """
    加载图像特征到shape=(n, 1024)的array里
    特征不需要shared，构建模型时传入后再shared
    """
    print(name + ' features ...')               # name = 'Image' or 'Text'
    fea_it = np.zeros((item_num + 1, it_size))  # 多出来一个，存放用于补齐用户购买序列的/实际不存在的item
    files = glob.glob(it_fea_path + '*.json')
    all_it_items = []
    for i, e in enumerate(files):
        print('\tread file {v1} / {v2}: {v3}'.format(v1=i, v2=len(files), v3=e.split('\\')[-1]))
        with open(e, 'r') as f:
            iff = json.load(f)
        all_it_items.extend([str(i) for i in iff.keys()])
        for key, val in iff.items():    # key是item的ID标号，val是list格式的
            try:
                # np.array(val)可转换为shape=(1024,)，正好是shape=(n, 2014)中的某一项
                fea_it[aliases_dict[str(key)]] = np.array(val)
            except KeyError:    # 图像特征多提了用户没购买商品的，比如用全集特征时，该处会报错, 因为此时dict里没有这个id
                # print('\tredundant image features: {val}, '.format(val=key),)    # 输出时不换行
                pass
        del iff
    ts1, ts2 = set(aliases_dict.keys()) - set(all_it_items), set(all_it_items) - set(aliases_dict.keys())
    print('\tWarning: These items do not have features. Set to zero. = {v1}'.format(v1=len(ts1)))
    print('\tWarning: These items\' features are redundant. No use. = {v1}'.format(v1=len(ts2)))
    return fea_it


# 不再用，效果一般。经过多次实验，表明先把data做成corrupted效果并不好. 在序列训练前做,效果更好更稳定。
# def get_corrupted_input_whole(inp, corruption_level):
#     # randomly set whole feature to zero
#     # denoising方式0：随机将某些图、文特征整体性置为0
#     # 比如原先图是(num, 1024)，那么0/1概率矩阵是(num, 1), 两个二维array直接相乘即可
#     randoms = binomial(
#         size=(inp.shape[0], 1),
#         n=1,
#         p=1 - corruption_level)
#     print('\tAttention: {v1}% whole features are randomly reset to zero.'.format(v1=corruption_level * 100))
#     return randoms * inp


# 不再用，这种耗时明显要多。比如Amazon的train由15min变为25min。
# def get_corrupted_input_position(inp, corruption_level):
#     # randomly set some positions in feature to zero
#     # denoising方式1：随机将特征的某些位置置为0，和dAE的原设置一样
#     # 比如原先图是(10, 1024)，那么0/1概率矩阵也是(10, 1024)
#     randoms = binomial(
#         size=inp.shape,
#         n=1,
#         p=1 - corruption_level)
#     print('\tAttention: {v1} % positions are randomly reset to zero.'.format(v1=corruption_level * 100))
#     return randoms * inp


def fun_data_buys_masks(all_user_buys, tail):
    # 将train/test中序列补全为最大长度，补的idx值=item_num. 为了能在theano里对其进行shared。
    # tail, 添加的。商品索引是0~item_num-1，所以该值[item_num]是没有对应商品实物的。
    us_lens = [len(ubuys) for ubuys in all_user_buys]
    len_max = max(us_lens)
    us_buys = [ubuys + tail * (len_max - le) for ubuys, le in zip(all_user_buys, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_buys, us_msks


def fun_random_neg_masks_tra(item_num, tras_mask):
    """
    从num件商品里随机抽取与每个用户的购买序列等长且不在已购买商品里的标号。后边补全的负样本用虚拟商品[item_num]
    """
    us_negs = []
    for utra in tras_mask:     # 每条用户序列
        unegs = []
        for i, e in enumerate(utra):
            if item_num == e:                    # 表示该购买以及之后的，都是用虚拟商品[item_num]来补全的
                unegs += [item_num] * (len(utra) - i)   # 购买序列里对应补全商品的负样本也用补全商品表示
                break
            j = random.randint(0, item_num - 1)  # 负样本在商品矩阵里的标号
            while j in utra:                     # 抽到的不是用户训练集里的。
                j = random.randint(0, item_num - 1)
            unegs += [j]
        us_negs.append(unegs)
    return us_negs


def fun_random_neg_masks_tes(item_num, tras_mask, tess_mask):
    """
    从num件商品里随机抽取与测试序列等长且不在训练序列、也不再测试序列里的标号
    """
    us_negs = []
    for utra, utes in zip(tras_mask, tess_mask):
        unegs = []
        for i, e in enumerate(utes):
            if item_num == e:                   # 尾部补全mask
                unegs += [item_num] * (len(utes) - i)
                break
            j = random.randint(0, item_num - 1)
            while j in utra or j in utes:         # 不在训练序列，也不在预测序列里。
                j = random.randint(0, item_num - 1)
            unegs += [j]
        us_negs.append(unegs)
    return us_negs


@exe_time  # 放到待调用函数的定义的上一行
def main():
    print('... load the dataset, and  no need to set shared.')


if '__main__' == __name__:
    main()
