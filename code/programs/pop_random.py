#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:       Qiang Cui:  <cuiqiang1990[at]hotmail.com>
# Descripton:   纯RNN、GRU程序
# Create Date:  2016-12-05 11:30:00
# Modify Date:  2016-00-00 00:00:00
# Modify Disp:

from __future__ import print_function
from collections import OrderedDict     # 按输入的顺序构建字典
import time
import numpy as np
import os
import random
from random import sample
from collections import Counter
from collections import defaultdict
from public.Global_Best import GlobalBest
from public.Valuate import fun_predict_pop_random
from public.Load_Data_by_time import load_data
from public.Load_Data_by_time import fun_data_buys_masks
from public.Load_Data_by_time import fun_random_neg_masks_tra, fun_random_neg_masks_tes
__docformat__ = 'restructedtext en'

WHOLE = 'E:/Projects/datasets/'
# WHOLE = '/home/cuiqiang/2_Datasets/'
# WHOLE = '/home/wushu/usr_cuiqiang/2_Datasets/'
# PATH_a5 = os.path.join(WHOLE, 'amazon_users_5_100/')
# PATH_t55 = os.path.join(WHOLE, 'taobao_users_5_100_5000hang/')
# PATH_t5 = os.path.join(WHOLE, 'taobao_users_5_100/')
PATH_a515 = os.path.join(WHOLE, 'amazon_users_5_100_items_5/')
PATH_t515 = os.path.join(WHOLE, 'taobao_users_5_100_items_5/')
PATH = PATH_a515     # 109上先试试t55，确保正常运行。


def exe_time(func):
    def new_func(*args, **args2):
        t0 = time.time()
        print("-- @%s, {%s} start" % (time.strftime("%X", time.localtime()), func.__name__))
        back = func(*args, **args2)
        print("-- @%s, {%s} end" % (time.strftime("%X", time.localtime()), func.__name__))
        print("-- @%.3fs taken for {%s}" % (time.time() - t0, func.__name__))
        return back
    return new_func


def compute_start_end(user_num, batch_size_test):
    """
    获取mini-batch的各个start_end(np.array类型，一组连续的数值)
    :param flag: 'train', 'test'
    :return: 各个start_end组成的list
    """
    size = batch_size_test * 10   # test: auc
    rest = (user_num % size) > 0   # 能整除：rest=0。不能整除：rest=1，则多出来一个小的batch
    n_batches = np.minimum(user_num // size + rest, user_num)
    batch_idxs = np.arange(n_batches, dtype=np.int32)
    starts_ends = []
    for bidx in batch_idxs:
        start = bidx * size
        end = np.minimum(start + size, user_num)   # 限制标号索引不能超过user_num
        start_end = np.arange(start, end, dtype=np.int32)
        starts_ends.append(start_end)
    return batch_idxs, starts_ends


def train_valid_or_test(p=None):
    """
    构建模型参数，加载数据
        把前80%分为6:2用作train和valid，来选择超参数, 不用去管剩下的20%.
        把前80%作为train，剩下的是test，把valid时学到的参数拿过来跑程序.
        valid和test部分，程序是一样的，区别在于送入的数据而已。
    :param p: 一个标示符，没啥用
    :return:
    """
    global PATH
    # 1. 建立各参数。要调整的地方都在 p 这了，其它函数都给写死。
    if not p:
        t = 't'                       # 写1就是valid, 写0就是test
        assert 't' == t or 'v' == t   # no other case
        p = OrderedDict(
            [
                ('dataset',             'user_buys.txt'),
                ('fea_image',           'normalized_features_image/'),
                ('fea_text',            'normalized_features_text/'),
                ('mode',                'test' if 't' == t else 'valid'),
                ('split',               [0.8, 1.0] if 't' == t else [0.6, 0.8]),  # valid: 6/2/2。test: 8/2.
                ('at_nums',             [10, 20, 30, 50]),  # 5， 15
                ('intervals',           [2, 10, 30]),       # 以次数2为间隔，分为10个区间. 计算auc/recall@30上的. 换为10

                ('batch_size_train',    4),     # size大了之后性能下降非常严重
                ('batch_size_test',     768),   # user*item矩阵太大，要多次计算。a5下亲测768最快。
            ])
        for e in p.items():
            print(e)
        assert 'valid' == p['mode'] or 'test' == p['mode']

    # 2. 加载数据
    # 因为train/set里每项的长度不等，无法转换为完全的(n, m)矩阵样式，所以shared会报错.
    [(user_num, item_num), aliases_dict,
     (test_i_cou, test_i_intervals_cumsum, test_i_cold_active),
     (tra_buys, tes_buys), (set_tra, set_tes)] = \
        load_data(os.path.join(PATH, p['dataset']), p['mode'], p['split'], p['intervals'])
    # 正样本加masks
    tra_buys_masks, tra_masks = fun_data_buys_masks(tra_buys, tail=[item_num])          # 预测时算用户表达用
    tes_buys_masks, tes_masks = fun_data_buys_masks(tes_buys, tail=[item_num])          # 预测时用
    # 负样本加masks
    # tra_buys_neg_masks = fun_random_neg_masks_tra(item_num, tra_buys_masks)   # 训练时用（逐条、mini-batch均可）
    tes_buys_neg_masks = fun_random_neg_masks_tes(item_num, tra_buys_masks, tes_buys_masks)   # 预测时用

    # --------------------------------------------------------------------------------------------------------------
    # 获得按购买次数由大到小排序的items, 出现次数相同的，随机排列。
    tra = []
    for buy in tra_buys:
        tra.extend(buy)
    train_i = set(tra)
    train_i_cou = dict(Counter(tra))                    # {item: num, }, 各个item出现的次数
    lst = defaultdict(list)
    for item, count in train_i_cou.items():
        lst[count].append(item)
    # 某个被购买次数(count)下各有哪些商品，商品数目是count。count越大，这些items越popular
    lst = list(lst.items())                             # [(num, [item1, item2, ...]), ]
    lst = list(sorted(lst, key=lambda x: x[0]))[::-1]   # 被购买次数多的，出现在首端
    sequence = []
    for count, items in lst:
        sequence.extend(random.sample(items, len(items)))   # 某个购买次数下的各商品，随机排列。

    def fun_judge_tes_and_neg(tes_mark_neg):
        tes, mark, tes_neg, _ = tes_mark_neg
        zero_one = []
        for idx, flag in enumerate(mark):
            if 0 == flag:
                zero_one.append(0)
            else:
                i, j = tes[idx], tes_neg[idx]
                if i in train_i and j in train_i:
                    zero_one.append(1 if train_i_cou[i] > train_i_cou[j] else 0)
                elif i in train_i and j not in train_i:
                    zero_one.append(1)
                elif i not in train_i and j in train_i:
                    zero_one.append(0)
                else:
                    zero_one.append(0)
        return zero_one     # 与mask等长的0/1序列。1表示用户买的商品比负样本更流行。

    # --------------------------------------------------------------------------------------------------------------
    print("\tPop ...")
    append = [[0] for _ in np.arange(len(tes_buys_masks))]
    all_upqs = np.apply_along_axis(     # 判断tes里的是否比tes_neg更流行
        func1d=fun_judge_tes_and_neg,
        axis=1,
        arr=np.array(zip(tes_buys_masks, tes_masks, tes_buys_neg_masks, append)))
    recom = sequence[:p['at_nums'][-1]]             # 每个用户都给推荐前100个最流行的
    all_ranks = np.array([recom for _ in np.arange(user_num)])

    # 存放最优数据。计算各种指标并输出。
    best = GlobalBest(at_nums=p['at_nums'], intervals=p['intervals'])
    fun_predict_pop_random(
        p, best, all_upqs, all_ranks,
        tes_buys_masks, tes_masks,
        test_i_cou, test_i_intervals_cumsum, test_i_cold_active)
    best.fun_print_best(epoch=0)   # 每次都只输出当前最优的结果

    # --------------------------------------------------------------------------------------------------------------
    print("\tRandom ...")
    all_upqs = None     # random的auc就是0.5，直接引用文献里的说法。
    seq_random = sample(sequence, len(sequence))    # 先把总序列打乱顺序。再每个用户都给随机推荐100个
    all_ranks = np.array([sample(seq_random, p['at_nums'][-1]) for _ in np.arange(user_num)])

    # 存放最优数据。计算各种指标并输出。
    best = GlobalBest(at_nums=p['at_nums'], intervals=p['intervals'])
    fun_predict_pop_random(
        p, best, all_upqs, all_ranks,
        tes_buys_masks, tes_masks,
        test_i_cou, test_i_intervals_cumsum, test_i_cold_active)
    best.fun_print_best(epoch=0)   # 每次都只输出当前最优的结果


@exe_time  # 放到待调用函数的定义的上一行
def main():
    train_valid_or_test()


if '__main__' == __name__:
    main()
