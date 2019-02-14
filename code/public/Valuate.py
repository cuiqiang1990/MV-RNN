#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:       Qiang Cui:  <cuiqiang1990[at]hotmail.com>
# Descripton:   construct the evaluation program
# Create Date:  2016-12-02 17:00:00
# Modify Date:  2016-00-00 00:00:00
# Modify Disp:

from __future__ import print_function
import time
import pandas as pd
import numpy as np
import math
import datetime
from numpy import maximum
from numpy import greater
from pandas import DataFrame
import os
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


# 结论: AUC不需要每个正样本都用剩余所有的items做负样本，会太耗时，且和每次都随机取一个负样本的效果没多大差别.
#      AUC更合理的算法是，一个正样本，对应k个负样本。


def fun_hit_zero_one(user_test_recom):
    """
    根据recom_list中item在test_lst里的出现情况生成与recom_list等长的0/1序列
    0表示推荐的item不在test里，1表示推荐的item在test里
    :param test_lst: 单个用户的test列表
    :param recom_lst: 推荐的列表
    :param test_mask: 单个用户的test列表对应的mask列表
    :return: 与recom_list等长的0/1序列。
    """
    test_lst, recom_lst, test_mask, _ = user_test_recom
    test_lst = test_lst[:np.sum(test_mask)]     # 取出来有效的user_test_list
    seq = []
    for e in recom_lst:
        if e in test_lst:       # 命中
            seq.append(1)
        else:                   # 没命中
            seq.append(0)
    return np.array(seq)


def fun_hit_auc_item_idx(test_upqs):
    """
    算auc时，已得出所有用户的所有test里正负计算，正的为true, 负的为false，得到那些为true的items_id
    -1表示推荐的item不在test里，idx表示推荐的item在test里所对应的item_index
    :param test_lst: 单个用户的test列表
    :param upq_lst: 正负计算得出来的正负列表
    :param test_mask: 单个用户的test列表对应的mask列表
    :return: 与upq_lst等长的标号为1对应的item_idx序列
    """
    test_lst, upq_lst, test_mask, _ = test_upqs
    idxs = []
    for i, e in enumerate(test_lst):
        if 1 == test_mask[i]:       # 这是前边那块有效的test
            if upq_lst[i]:
                idxs.append(e)
            else:
                idxs.append(-1)
        else:                       # test后边补0的位置。
            idxs.append(-1)
    return np.array(idxs)


def fun_hit_recall_item_idx(user_test_recom):
    """
    根据recom_list中item在test_lst里的出现情况生成与recom_list等长的-1/idx序列
    -1表示推荐的item不在test里，idx表示推荐的item在test里所对应的item_index
    :param test_lst: 单个用户的test列表
    :param recom_lst: 推荐的列表
    :param test_mask: 单个用户的test列表对应的mask列表
    :return: 与recom_list等长的-1/idx序列。得到那些命中item的idx。
    """
    test_lst, recom_lst, test_mask, _ = user_test_recom
    test_lst = test_lst[:np.sum(test_mask)]     # 取出来有效的user_test_list
    idxs = []
    for e in recom_lst:
        if e in test_lst:       # 命中
            idxs.append(e)
        else:                   # 没命中。不能标为0，因为有一个item的别名是0
            idxs.append(-1)     # 不要标为np.nan，会有奇葩的错误
    return np.array(idxs)


def fun_item_idx_to_intervals(all_item_idxs, test_i_cou, p_inters):
    """
    根据标号矩阵找出标号为1的item_id，根据其在test里的出现次数，计算分别在各个区间里出现了多少交易量
    :param all_item_idxs: 命中的item_index
    :param test_i_cou: test里items出现次数的dict
    :param p_inters: [2, 10, 30]
    :return: 每个区间里各有多少交易量
    """
    intervals = np.array([0 for _ in range(p_inters[1])])     # 有这么些个区间
    hit_item_idxs = [idx for u_idxs in all_item_idxs
                     for idx in u_idxs if -1 != idx]     # 先取出来那些item_id
    for idx in hit_item_idxs:                                       # 把item_id分到各个区间
        inter = int(math.ceil(1.0 * test_i_cou[idx] / p_inters[0])) - 1
        if inter >= p_inters[1] - 1:
            inter = -1
        intervals[inter] += 1
    return intervals


def fun_evaluate_map(user_test_recom_zero_one):
    """
    计算map。所得是单个用户test的，最后所有用户的求和取平均
    :param test_lst: 单个用户的test集
    :param zero_one: 0/1序列
    :param test_mask: 单个用户的test列表对应的mask列表
    :return:
    """
    test_lst, zero_one, test_mask, _ = user_test_recom_zero_one
    test_lst = test_lst[:np.sum(test_mask)]

    zero_one = np.array(zero_one)
    if 0 == sum(zero_one):    # 没有命中的
        return 0.0
    zero_one_cum = zero_one.cumsum()                # precision要算累计命中
    zero_one_cum *= zero_one                        # 取出命中为1的那些，其余位置得0
    idxs = list(np.nonzero(zero_one_cum))[0]        # 得到(n,)类型的非零的索引array
    s = 0.0
    for idx in idxs:
        s += 1.0 * zero_one_cum[idx] / (idx + 1)
    return s / len(test_lst)


def fun_evaluate_ndcg(user_test_recom_zero_one):
    """
    计算ndcg。所得是单个用户test的，最后所有用户的求和取平均
    :param test_lst: 单个用户的test集
    :param zero_one: 0/1序列
    :param test_mask: 单个用户的test列表对应的mask列表
    :return:
    """
    test_lst, zero_one, test_mask, _ = user_test_recom_zero_one
    test_lst = test_lst[:np.sum(test_mask)]

    zero_one = np.array(zero_one)
    if 0 == sum(zero_one):    # 没有命中的
        return 0.0
    s = 0.0
    idxs = list(np.nonzero(zero_one))[0]
    for idx in idxs:
        s += 1.0 / np.log2(idx + 2)
    m = 0.0
    length = min(len(test_lst), len(zero_one))      # 序列短的，都命中为1，此时是最优情况
    for idx in range(length):
        m += 1.0 / np.log2(idx + 2)
    return s / m


def fun_idxs_of_max_n_score(user_scores_to_all_items, top_k):
    # a = np.asarray([5, 4, 3, 2, 1, 0])  # 各item_idx的得分
    # b = np.argpartition(a, -3)[-3:]     # 前3个得分最大的idx
    # t = [0, 1, 2]
    # print(set(b) == set(t))     # True。与顺序无关。

    # 从一个向量里找到前n个大数所对应的index
    return np.argpartition(user_scores_to_all_items, -top_k)[-top_k:]


def fun_sort_idxs_max_to_min(user_max_n_idxs_scores):
    # a = np.asarray([5, 4, 3, 2, 1, 0])  # 各item_idx的得分
    # b = np.argpartition(a, -3)[-3:]     # 前3个得分最大的idx
    # c = b[np.argsort(a[b])][::-1]       # 最左侧的idx对应的得分最大。
    # t = [0, 1, 2]
    # print(list(c) == t)         # True。与顺序有关。

    # 按照前n个大数index对应的具体值由大到小排序，即最左侧的index对应原得分值的最大实数值
    # 就是生成的推荐items列表里，最左侧的item_idx是得分最高的
    idxs, scores = user_max_n_idxs_scores           # idxs是n个得分最大的items，scores是所有items的得分。
    return idxs[np.argsort(scores[idxs])][::-1]     # idxs按照对应得分由大到小排列


def fun_predict_auc_recall_map_ndcg(
        p, model, best, epoch, starts_ends_auc, starts_ends_tes,
        tes_buys_masks, tes_masks,
        test_i_cou, test_i_intervals_cumsum, test_i_cold_active):
    # ------------------------------------------------------------------------------------------------------------------
    # 注意：当zip里的所有子项tes_buys_masks, all_upqs维度一致时，也就是子项的每行每列长度都一样。
    #      zip后的arr会变成三维矩阵，扫描axis=1会出错得到的一行实际上是2d array，所以后边单独加一列 append 避免该问题。
    append = [[0] for _ in np.arange(len(tes_buys_masks))]

    # ------------------------------------------------------------------------------------------------------------------
    # auc
    all_upqs = []   # shape = tes_buys_masks.shape
    for start_end in starts_ends_auc:
        sub_all_upqs = model.compute_sub_auc_preference(start_end)
        all_upqs.extend(sub_all_upqs)
    all_upqs = np.asarray(all_upqs)
    # 计算：auc_intervals的当前epoch值。
    auc = 1.0 * np.sum(all_upqs) / np.sum(tes_masks)    # 全部items。sum(tes_masks)是按列加和，不对。
    # 找出all_upqs里标为true的那些items的标号
    auc_item_idxs = np.apply_along_axis(    # 找出all_upqs里标为true的那些items的标号
        func1d=fun_hit_auc_item_idx,
        axis=1,
        arr=np.array(zip(tes_buys_masks, all_upqs, tes_masks, append)))
    auc_intervals = fun_item_idx_to_intervals(
        auc_item_idxs,
        test_i_cou,
        p['intervals'])

    # 保存：保存auc的最佳值
    auc_cold_active = np.array([sum(auc_intervals[:2]), sum(auc_intervals[2:])])
    auc_intervals_cumsum = auc_intervals.cumsum()   # 最后一个值和AUC总值相等，就表明auc_intervals没有计算错误。
    if auc > best.best_auc:
        best.best_auc = auc
        best.best_epoch_auc = epoch
        best.best_auc_cold_active = 1.0 * auc_cold_active / test_i_cold_active
        best.best_auc_intervals_cumsum = 1.0 * auc_intervals_cumsum / test_i_intervals_cumsum

    # ------------------------------------------------------------------------------------------------------------------
    # recall, map, ndcg
    at_nums = p['at_nums']          # [5, 10, 15, 20, 30, 50]
    ranges = range(len(at_nums))

    # 计算：所有用户对所有商品预测得分的前50个。
    # 不会预测出来添加的那个虚拟商品，因为先把它从item表达里去掉
    # 注意矩阵形式的索引 all_scores[0, rank]：表示all_scores的各行里取的各个列值是rank里的各项
    all_ranks = []  # shape=(n, 50)
    for start_end in starts_ends_tes:
        sub_all_scores = model.compute_sub_all_scores(start_end)  # shape=(sub_n_user, n_item)
        sub_score_ranks = np.apply_along_axis(
            func1d=fun_idxs_of_max_n_score,
            axis=1,
            arr=sub_all_scores,
            top_k=at_nums[-1])
        sub_all_ranks = np.apply_along_axis(
            func1d=fun_sort_idxs_max_to_min,
            axis=1,
            arr=np.array(zip(sub_score_ranks, sub_all_scores)))
        all_ranks.extend(sub_all_ranks)
        del sub_all_scores
    all_ranks = np.asarray(all_ranks)

    # 计算：recall、map、ndcg当前epoch的值
    arr = np.array([0.0 for _ in ranges])
    recall, precis, f1scor, map_, ndcg = arr.copy(), arr.copy(), arr.copy(), arr.copy(), arr.copy()
    hits, denominator_recalls = arr.copy(), np.sum(tes_masks)  # recall的分母，要预测这么些items
    for k in ranges:                            # 每次考察某个at值下的命中情况
        recoms = all_ranks[:, :at_nums[k]]      # 向每名user推荐这些
        # 逐行，得到recom_lst在test_lst里的命中情况，返回与recom_lst等长的0/1序列，1表示预测的该item在user_test里
        all_zero_ones = np.apply_along_axis(
            func1d=fun_hit_zero_one,
            axis=1,
            arr=np.array(zip(tes_buys_masks, recoms, tes_masks, append)))   # shape=(n_user, at_nums[k])
        hits[k] = np.sum(all_zero_ones)
        recall[k] = 1.0 * np.sum(all_zero_ones) / denominator_recalls
        precis[k] = 1.0 * np.sum(all_zero_ones) / (at_nums[k] * len(all_zero_ones))
        f1scor[k] = 2.0 * recall[k] * precis[k] / (recall[k] + precis[k])
        all_maps = np.apply_along_axis(
            func1d=fun_evaluate_map,
            axis=1,
            arr=np.array(zip(tes_buys_masks, all_zero_ones, tes_masks, append)))
        map_[k] = np.mean(all_maps)
        all_ndcgs = np.apply_along_axis(
            func1d=fun_evaluate_ndcg,
            axis=1,
            arr=np.array(zip(tes_buys_masks, all_zero_ones, tes_masks, append)))
        ndcg[k] = np.mean(all_ndcgs)

    # 计算：hit_intervals的当前epoch值。取recall@30
    recoms = all_ranks[:, :p['intervals'][2]]
    recall_item_idxs = np.apply_along_axis(    # 找出recoms里出现在test里的那些items的标号
        func1d=fun_hit_recall_item_idx,
        axis=1,
        arr=np.array(zip(tes_buys_masks, recoms, tes_masks, append)))
    hit_intervals = fun_item_idx_to_intervals(
        recall_item_idxs,
        test_i_cou,
        p['intervals'])

    # 保存：recall/map/ndcg/hit_interval的最佳值
    # recall and intervals
    hit_cold_active = np.array([sum(hit_intervals[:2]), sum(hit_intervals[2:])])
    hit_intervals_cumsum = hit_intervals.cumsum()
    for k in ranges:
        if recall[k] > best.best_recall[k]:
            best.best_recall[k] = recall[k]
            best.best_epoch_recall[k] = epoch
            if p['intervals'][2] == at_nums[k]:         # recall@30的区间划分
                best.best_recalls_cold_active = 1.0 * hit_cold_active / test_i_cold_active
                best.best_recalls_intervals_cumsum = 1.0 * hit_intervals_cumsum / test_i_intervals_cumsum
    # map and ndcg
    for k in ranges:
        if precis[k] > best.best_precis[k]:
            best.best_precis[k] = precis[k]
            best.best_epoch_precis[k] = epoch
        if f1scor[k] > best.best_f1scor[k]:
            best.best_f1scor[k] = f1scor[k]
            best.best_epoch_f1scor[k] = epoch
        if map_[k] > best.best_map[k]:
            best.best_map[k] = map_[k]
            best.best_epoch_map[k] = epoch
        if ndcg[k] > best.best_ndcg[k]:
            best.best_ndcg[k] = ndcg[k]
            best.best_epoch_ndcg[k] = epoch
    del all_upqs, all_ranks


def fun_predict_pop_random(
        p, best, all_upqs, all_ranks,
        tes_buys_masks, tes_masks,
        test_i_cou, test_i_intervals_cumsum, test_i_cold_active):

    append = [[0] for _ in np.arange(len(tes_buys_masks))]

    # 计算：AUC。
    if all_upqs is not None:
        auc = 1.0 * np.sum(all_upqs) / np.sum(tes_masks)    # 全部items

        auc_item_idxs = np.apply_along_axis(    # 找出all_upqs里标为true的那些items的标号
            func1d=fun_hit_auc_item_idx,
            axis=1,
            arr=np.array(zip(tes_buys_masks, all_upqs, tes_masks, append)))
        auc_intervals = fun_item_idx_to_intervals(
            auc_item_idxs,
            test_i_cou,
            p['intervals'])
        # 保存auc
        auc_cold_active = np.array([sum(auc_intervals[:2]), sum(auc_intervals[2:])])
        auc_intervals_cumsum = auc_intervals.cumsum()
        best.best_auc = auc
        best.best_auc_cold_active = 1.0 * auc_cold_active / test_i_cold_active
        best.best_auc_intervals_cumsum = 1.0 * auc_intervals_cumsum / test_i_intervals_cumsum

    at_nums = p['at_nums']
    ranges = range(len(at_nums))
    # 计算：recall、map、ndcg当前epoch的值
    arr = np.array([0.0 for _ in ranges])
    recall, precis, f1scor, map_, ndcg = arr.copy(), arr.copy(), arr.copy(), arr.copy(), arr.copy()
    hits, denominator_recalls = arr.copy(), np.sum(tes_masks)  # recall的分母，要预测这么些items
    for k in ranges:                            # 每次考察某个at值下的命中情况
        recoms = all_ranks[:, :at_nums[k]]      # 向每名user推荐这些
        # 得到predict的每行在test_buys每行里的命中情况，返回与predict等长的0/1序列，1表示预测的该item在user_test里
        all_zero_ones = np.apply_along_axis(
            func1d=fun_hit_zero_one,
            axis=1,
            arr=np.array(zip(tes_buys_masks, recoms, tes_masks, append)))   # shape=(n_user, at_nums[k])
        hits[k] = np.sum(all_zero_ones)
        recall[k] = 1.0 * np.sum(all_zero_ones) / denominator_recalls
        precis[k] = 1.0 * np.sum(all_zero_ones) / (at_nums[k] * len(all_zero_ones))
        f1scor[k] = 2.0 * recall[k] * precis[k] / (recall[k] + precis[k])
        all_maps = np.apply_along_axis(
            func1d=fun_evaluate_map,
            axis=1,
            arr=np.array(zip(tes_buys_masks, all_zero_ones, tes_masks, append)))
        map_[k] = np.mean(all_maps)
        all_ndcgs = np.apply_along_axis(
            func1d=fun_evaluate_ndcg,
            axis=1,
            arr=np.array(zip(tes_buys_masks, all_zero_ones, tes_masks, append)))
        ndcg[k] = np.mean(all_ndcgs)

    # 计算：hit_intervals的当前epoch值。
    recoms = all_ranks[:, :p['intervals'][2]]
    hit_item_idxs = np.apply_along_axis(    # 找出recoms里出现在test里的那些items的标号
        func1d=fun_hit_recall_item_idx,
        axis=1,
        arr=np.array(zip(tes_buys_masks, recoms, tes_masks, append)))
    hit_intervals = fun_item_idx_to_intervals(
        hit_item_idxs,
        test_i_cou,
        p['intervals'])

    # 保存：保存auc/recall/map/ndcg/hit_interval的最佳值
    # recall and intervals
    hit_cold_active = np.array([sum(hit_intervals[:2]), sum(hit_intervals[2:])])
    hit_intervals_cumsum = hit_intervals.cumsum()
    for k in ranges:
        if recall[k] > best.best_recall[k]:
            best.best_recall[k] = recall[k]
            if p['intervals'][2] == at_nums[k]:         # recall@30的区间划分
                best.best_recalls_cold_active = 1.0 * hit_cold_active / test_i_cold_active
                best.best_recalls_intervals_cumsum = 1.0 * hit_intervals_cumsum / test_i_intervals_cumsum
    # map and ndcg
    for k in ranges:
        best.best_precis[k] = precis[k]
        best.best_f1scor[k] = f1scor[k]
        best.best_map[k] = map_[k]
        best.best_ndcg[k] = ndcg[k]


# 保存训练结束后每个item_id对应学好的特征表达。30轮迭代，最后一轮输出结果即可。
def fun_save_final_item_feas(
        path, model_name, model,
        aliases):
    # 调用示例
    # if epoch == p['epochs'] - 1 and 20 == size:        # 保存20维时的最后一次迭代的item_id、item表达
    #     path = os.path.join(
    #         os.path.split(__file__)[0],     # 程序文件所在目录
    #         '..', 'Results_trained_item_feas',
    #         PATH.split('/')[-2])        # 数据库名
    #     fun_save_final_item_feas(
    #         path,                                   # 创建目录并保存
    #         aliases_dict, all_its,
    #         model_name)

    # 特征进一步分为5个文件：latent, visual, textual, multi-modal, [latent;multi-modal]
    lt, fi, ft, ei, vt = model.lt.eval(), model.fi.eval(), model.ft.eval(), model.ei.eval(), model.vt.eval()
    fi = np.dot(fi, ei.T)   # shape=(n_item+1, 20)
    ft = np.dot(ft, vt.T)
    mm = fi + ft
    items = np.concatenate((lt, mm), axis=1)  # shape=(n_item+1, 40)
    heads = ['latent', 'visual', 'textual', 'multi-modal(visual+textual)', '[latent;multi-modal]']
    all_its = [lt, fi, ft, mm, items]

    # 直接把aliases和all_its合到一块，key是原始item_id，value是特征表达
    keys = aliases.keys()   # keys是图片的id，就是原始item_id
    feas = []
    for i in range(len(all_its)):
        tmp_save = []
        one_feas = all_its[i]
        for item_id in keys:
            fea = one_feas[aliases[item_id]]    # 最后特征比aliases多着一个用来标mask的，这个不会被保存。
            tmp_save.append(','.join([str(i) for i in fea]))
        feas.append(tmp_save)

    cols = ['item_id'] + heads
    df = DataFrame({cols[0]: [i for i in keys],     # amazon里的item_id是符号，不是数字，不能用int(i)
                    cols[1]: feas[0],
                    cols[2]: feas[1],
                    cols[3]: feas[2],
                    cols[4]: feas[3],
                    cols[5]: feas[4]})
    df.sort_values(by=['item_id'], ascending=True, inplace=True)
    # 创建路径、文件名
    if os.path.exists(path):
        print('\t\tdir exists: {v1}'.format(v1=path))
    else:
        os.makedirs(path)
        print('\t\tdir is made: {v1}'.format(v1=path))
    now = datetime.datetime.now()
    now = now.strftime("_%Y%m%d_%H%M%S")
    path_save_file = os.path.join(
        path,
        'final_feature_of_all_items_20d_on_' + model_name + now + '.txt')

    df.to_csv(path_save_file, sep=' ', index=False, columns=cols)
    print('\t\tFinal representations of all items are saved.')


def fun_acquire_fil_para(model_name, p):
    # 获取要保存的文件的文件头（各种参数）
    if 'HcaGru' in model_name:
        winx, winh = p['window_x'], p['window_h']
    else:
        winx, winh = 0, 0
    alpha_lambda = [p['alpha'], p['lambda']]
    batch_sizes = [p['batch_size_train'], p['batch_size_test']]
    size, epoch, at_nums = p['latent_size'], p['epochs'], p['at_nums']
    fil_para = \
        '\n' + model_name + \
        '\n\t' + 'winx, winh = {v1}'.format(v1=', '.join([str(i) for i in [winx, winh]])) + \
        '\n\t' + 'alpha, lambda = {v1}'.format(v1=', '.join([str(i) for i in alpha_lambda])) + \
        '\n\t' + 'batch_size train, test = {v1}'.format(v1=', '.join([str(i) for i in batch_sizes])) + \
        '\n\t' + 'size, epoch, at_nums = {v1}d, {v2}, top-{v3}'.format(v1=size, v2=epoch, v3=at_nums) + \
        '\n'
    return fil_para


# 所有模型都可以用。
def fun_save_best_and_losses(
        path, model_name, epoch, p, best, losses):
    # 建立目录、文件名
    if os.path.exists(path):
        print('\t\tdir exists: {v1}'.format(v1=path))
    else:
        os.makedirs(path)
        print('\t\tdir is made: {v1}'.format(v1=path))
    size = p['latent_size'][0]
    size = '{v1}d_'.format(v1=size)
    fil_name = size + model_name + '.txt'       # 一个模型的，都放入一个文件名里。
    fil = os.path.join(path, fil_name)
    print('\t\tfile name: {v1}'.format(v1=fil_name))

    # 建立内容并保存
    f = open(fil, 'a')
    fil_para = fun_acquire_fil_para(model_name, p)
    fil_best = best.fun_obtain_best(epoch)  # 里面有输出best时的时间。
    fil_loss = \
        '\n\tLosses: ' + \
        '\n\t\t[{v1}]'.format(v1=', '.join(losses))
    f.write(fil_para)
    f.write(fil_best)
    f.write(fil_loss)
    f.write('\n')
    f.close()


# 用于attention模型。
def fun_save_atts(
        path, model_name, epoch, p, best,
        all_ats, path_dataset):
    # atts_name 举例：winx = 4, winh = 5，则保存5个h的权重，且每个h下有4个x的权重。
    # 对 winh 个 h 做 context，而每个 h 下有 winx 个输入做 context，所以(winh, winx)。
    # atts_name = [
    #     x1,x2,x3,x4;
    #     x2,x3,x4,x5;
    #     x3,x4,x5,x6;
    #     x4,x5,x6,x7;
    #     x5,x6,x7,x8;      # winh 组(winx, )的输入权重
    #     h1,h2,h3,h4,h5]   #  1   组(winh, )的隐层权重
    winx, winh = p['window_x'], p['window_h']
    windows = '_x{v1}_h{v2}'.format(v1=winx, v2=winh)

    # 建立目录、文件名
    if os.path.exists(path):
        print('\t\tdir exists: {v1}'.format(v1=path))
    else:
        os.makedirs(path)
        print('\t\tdir is made: {v1}'.format(v1=path))
    size = p['latent_size'][0]
    size = '{v1}d_'.format(v1=size)
    fil_name = size + model_name + windows + '.txt'       # 一个模型的，都放一块
    fil = os.path.join(path, fil_name)
    print('\t\tfile name: {v1}'.format(v1=fil_name))

    # 建立attention权重的名称
    atts_name, atts_name_h = [], []         # 建立权重的名字
    for i in range(1, winh + 1):            # 1,2,3,4,5
        tmp = []
        for j in range(winx):               # 0,1,2,3,4
            tmp.append('x' + str(i + j))    # x1,x2,x3,x4; x2,x3,x4,x5;
        atts_name.append(','.join(tmp))
        atts_name_h.append('h' + str(i))    # h1,h2,h3,h4,h5
    atts_name.append(','.join(atts_name_h))
    atts_name = '; '.join(atts_name)

    # 建立要保存的内容。头几列，后边的列就都是权重了。
    cols = ['buy_times', 'buy_different', 'user_id', atts_name]
    buys = pd.read_csv(path_dataset, sep=' ')   # 读进来后，是按照源文档里的顺序的。

    # 权重划分：每组内部用','隔开，组间用'; '隔开
    def truncate3(uatts_append):
        uatts, _ = uatts_append
        at_h = uatts[-winh:]            # 后边winh个
        at_x_len = len(uatts) - winh    # 这些是winh组winx的总长度，但序列太短的用户不够winh组。
        at_x = [[] for _ in np.arange(at_x_len // winx)]
        for k, att in enumerate(uatts[:-winh]):       # 前边winh组winx的所有的权重
            num = k // winx             # 得到每组权重的区间index
            at_x[num].append(att)
        at_x.append(at_h)
        at_x = [','.join(['%0.3f' % k for k in sub]) for sub in at_x]   # 每组内部是','隔开
        at_x = '; '.join(at_x)
        return at_x
    append = [[0] for _ in np.arange(len(all_ats))]
    all_ats_str = np.apply_along_axis(
        func1d=truncate3,
        axis=1,
        arr=np.asarray(zip(all_ats, append)))

    # 逐行读取、保存
    f = open(fil, 'a')
    fil_para = fun_acquire_fil_para(model_name, p)
    fil_best = best.fun_obtain_best(epoch)
    fil_cols = '\n' + ' '.join(cols)
    f.write(fil_para)
    f.write(fil_best)
    f.write(fil_cols)
    for a, b, c, e in \
            zip(list(buys[cols[0]]), list(buys[cols[1]]), list(buys[cols[2]]), all_ats_str):
        f.write('\n' + ' '.join([str(i) for i in [a, b, c, e]]))
    f.write('\n')
    f.close()


# 不再用。每个正样本都用剩余所有的items做负样本会太耗时，且和每次都随机取一个负样本的效果没多大差别。
# def fun_predict_auc_full(
#         best, epoch,
#         all_its, all_hus,
#         user_num, item_num,
#         tra_buys, tes_buys):
#     # test里每个正样本与剩余的所有负样本做差
#     # 开销最大的是每个用户都要与所有商品算乘积得到其对items的偏好
#     def fun_all_uijs(u):
#         u = u[0]
#         hu = all_hus[u]                     # 用户表达
#         xis = all_its[tes_buys[u]]          # shape=(n_xis, 20)
#         xjs = all_its[list(all_items - set(tra_buys[u]) - set(tes_buys[u]))]     # shape=(n_xjs, 20)
#         hu_xis = np.dot(xis, hu)            # shape=(n_xis, )，先把喜好程度计算好
#         hu_xjs = np.dot(xjs, hu)            # shape=(n_xjs, )
#         tmp = 0
#         for i in hu_xis:        # 这里写成并行没啥意义。不仅耗时可忽略，而且每个用户都必行，就已经达到cpu上限了。
#             ijs = i - hu_xjs
#             ijs = greater(ijs, 0)
#             tmp += np.sum(ijs)
#         return 1.0 * tmp / len(xjs) / len(xis)
#
#     all_items = set(range(item_num))
#     uidxs = np.arange(user_num)
#     # 最好还是做成mini-batch的，每次并行的个数 = cpu虚拟核数。一下子全部并行，效率不高的。
#     zimu = np.apply_along_axis(
#         func1d=fun_all_uijs,
#         axis=1,
#         arr=np.asarray(uidxs).reshape((user_num, 1)))
#     auc = np.average(zimu)
#
#     # 保存：保存auc的最佳值
#     if auc > best.best_auc_full:
#         best.best_auc_full = auc
#         best.best_epoch_auc_full = epoch


@exe_time  # 放到待调用函数的定义的上一行
def main():
    print('... construct the evaluation program')


if '__main__' == __name__:
    main()
