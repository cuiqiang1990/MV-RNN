#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:       Qiang Cui:  <cuiqiang1990[at]hotmail.com>
# Descripton:   
# Create Date:  2016-00-00 00:00:00
# Modify Date:  2016-00-00 00:00:00
# Modify Disp:

import time
import numpy as np
import os
import datetime


def exe_time(func):
    def new_func(*args, **args2):
        t0 = time.time()
        print("-- @%s, {%s} start" % (time.strftime("%X", time.localtime()), func.__name__))
        back = func(*args, **args2)
        print("-- @%s, {%s} end" % (time.strftime("%X", time.localtime()), func.__name__))
        print("-- @%.3fs taken for {%s}" % (time.time() - t0, func.__name__))
        return back
    return new_func


class GlobalBest(object):
    # 全局评价指标放在这里。保存最优的值、对应的epoch
    def __init__(self, at_nums, intervals=None):
        """
        :param at_nums:     [5, 10, 15, 20, 30, 50]
        :param intervals:   [2, 10, 30]
        :return:
        """
        self.intervals = intervals
        ranges = np.arange(len(at_nums))
        val_flo = np.array([0.0 for _ in ranges])
        epo_int = np.array([0 for _ in ranges])
        inters = np.array([0.0 for _ in np.arange(intervals[1])])   # 10个区间
        cold_active = np.array([0.0, 0.0])

        self.best_auc = 0.0
        self.best_recall = val_flo.copy()
        self.best_precis = val_flo.copy()
        self.best_f1scor = val_flo.copy()  # f1 = 2PR/(P+R)
        self.best_map = val_flo.copy()
        self.best_ndcg = val_flo.copy()

        self.best_epoch_auc = 0
        self.best_epoch_recall = epo_int.copy()
        self.best_epoch_precis = epo_int.copy()
        self.best_epoch_f1scor = epo_int.copy()
        self.best_epoch_map = epo_int.copy()
        self.best_epoch_ndcg = epo_int.copy()

        self.best_auc_intervals_cumsum = inters.copy()
        self.best_auc_cold_active = cold_active.copy()

        self.best_recalls_intervals_cumsum = inters.copy()
        self.best_recalls_cold_active = cold_active.copy()

    def fun_obtain_best(self, epoch):
        """
        :param epoch:
        :return: 由最优值组成的字符串
        """
        def truncate4(x):
            """
            把输入截断为六位小数
            :param x:
            :return: 返回一个字符串而不是list里单项是字符串，
            这样打印的时候就严格是小数位4维，并且没有字符串的起末标识''
            """
            return ', '.join(['%0.4f' % i for i in x])
        amp = 100
        one = '\t'
        two = one * 2
        now = datetime.datetime.now()
        a = one + '-----------------------------------------------------------------'
        # 指标值、最佳的epoch
        b = one + 'All values is the "best * {v1}" on epoch {v2}: | {v3}'\
            .format(v1=amp, v2=epoch, v3=now.strftime("%Y.%m.%d %H:%M:%S"))     # 这个时间要保留。
        c = two + 'AUC    = [{val}], '.format(val=truncate4([self.best_auc * amp])) + \
            two + '{val}'.format(val=[self.best_epoch_auc])
        d = two + 'Recall = [{val}], '.format(val=truncate4(self.best_recall * amp)) + \
            two + '{val}'.format(val=self.best_epoch_recall)
        # e = two + 'Precis = [{val}], '.format(val=truncate4(self.best_precis * amp)) + \
        #     two + '{val}'.format(val=self.best_epoch_precis)    # 不输出precision
        # f = two + 'F1scor = [{val}], '.format(val=truncate4(self.best_f1scor * amp)) + \
        #     two + '{val}'.format(val=self.best_epoch_f1scor)
        g = two + 'MAP    = [{val}], '.format(val=truncate4(self.best_map * amp)) + \
            two + '{val}'.format(val=self.best_epoch_map)
        h = two + 'NDCG   = [{val}], '.format(val=truncate4(self.best_ndcg * amp)) + \
            two + '{val}'.format(val=self.best_epoch_ndcg)
        # 分区间考虑。调参时去掉这块。
        i = one + 'cold_active | Intervals_cumsum:'
        j = two + 'AUC    = [{v1}], [{v2}]'. \
            format(v1=truncate4(self.best_auc_cold_active * amp),
                   v2=truncate4(self.best_auc_intervals_cumsum * amp))
        k = two + 'Recall@{v1} = [{v2}], [{v3}]'. \
            format(v1=self.intervals[2],
                   v2=truncate4(self.best_recalls_cold_active * amp),
                   v3=truncate4(self.best_recalls_intervals_cumsum * amp))
        return '\n'.join([a, b, c, d, g, h, i, j, k])

    def fun_print_best(self, epoch):
        # 输出最优值
        print(self.fun_obtain_best(epoch))


@exe_time  # 放到待调用函数的定义的上一行
def main():
    obj = GlobalBest(
        at_nums=[5, 10, 20, 30, 50, 100],
        intervals=[2, 10, 30])
    print('创建类对象后，读取实例变量值')
    print(obj.best_auc)     # 建立类对象后，可读取实例变量

    obj.best_auc = 70.3
    print('直接复制操作修改实例变量，再读取查看效果')
    print(obj.best_auc)     # 实例变量可直接修改


if '__main__' == __name__:
    main()


