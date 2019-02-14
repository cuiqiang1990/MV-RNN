#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:       Qiang Cui:  <cuiqiang1990[at]hotmail.com>
# Descripton:   construct the class RNN, GRU
# Create Date:  2016-12-02 17:00:00
# Modify Date:  2016-00-00 00:00:00
# Modify Disp:

from __future__ import print_function
import time
import numpy as np
from numpy.random import uniform
import theano
import theano.tensor as T
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh
from theano.tensor.extra_ops import Unique
from LSTM import LstmBasic
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


# 以此为基础，写的p-RNN
# 输出时：h*v → sigmoid(T.sum(h*(vp-vq), axis=1))
# 预测时：h*v → np.dot(h, v.T)
# ======================================================================================================================
class Lstm4Rec(LstmBasic):
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_in, n_hidden):
        super(Lstm4Rec, self).__init__(train, test, alpha_lambda, n_user, n_item, n_in, n_hidden)
        rang = 0.5
        # input
        lt = uniform(-rang, rang, (n_item + 1, n_in))   # 多出来一个(填充符)，存放用于补齐用户购买序列/实际不存在的item
        self.lt = theano.shared(borrow=True, value=lt.astype(theano.config.floatX))
        # hidden state
        uix = uniform(-rang, rang, (4, n_hidden, n_in))
        whx = uniform(-rang, rang, (4, n_hidden, n_hidden))
        c0x = np.zeros((n_hidden, ), dtype=theano.config.floatX)
        h0x = np.zeros((n_hidden, ), dtype=theano.config.floatX)
        bix = np.zeros((4, n_hidden), dtype=theano.config.floatX)
        self.uix = theano.shared(borrow=True, value=uix.astype(theano.config.floatX))
        self.whx = theano.shared(borrow=True, value=whx.astype(theano.config.floatX))
        self.c0x = theano.shared(borrow=True, value=c0x)
        self.h0x = theano.shared(borrow=True, value=h0x)
        self.bix = theano.shared(borrow=True, value=bix)
        # output
        vyx = uniform(-rang, rang, (n_item + 1, n_hidden))       # 和lt一样。
        self.vyx = theano.shared(borrow=True, value=vyx.astype(theano.config.floatX))
        # square loss
        self.paramsx = [self.uix, self.whx, self.bix]       # self.lt单独进行更新。
        self.l2_sqrx = (
            T.sum(self.lt ** 2) + T.sum(self.vyx ** 2) +
            T.sum([T.sum(param ** 2) for param in self.paramsx]))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * self.l2_sqrx)
        self.__theano_trainx__(n_in, n_hidden)
        self.__theano_predictx__(n_in, n_hidden)

    def __theano_trainx__(self, n_in, n_hidden):
        """
        训练阶段跑一遍训练序列
        """
        # self.alpha_lambda = ['alpha', 'lambda', 'fea_random_zero']
        uix, whx = self.uix, self.whx

        tra_mask = T.imatrix()                          # shape=(n, 157)
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))     # 获取mini-batch里各序列的长度最大值作为seq_length
        mask = tra_mask.T                               # shape=(157, n)

        c0x = T.alloc(self.c0x, actual_batch_size, n_hidden)      # shape=(n, 20)
        h0x = T.alloc(self.h0x, actual_batch_size, n_hidden)      # shape=(n, 40)
        bix = T.alloc(self.bix, actual_batch_size, 4, n_hidden)   # shape=(n, 3, 40), n_hidden放在最后
        bix = bix.dimshuffle(1, 2, 0)                             # shape=(3, 40, n)

        # 输入端：只输入购买的商品即可。
        pidxs, qidxs = T.imatrix(), T.imatrix()     # TensorType(int32, matrix)
        ixps = self.lt[pidxs]       # shape((actual_batch_size, seq_length, n_in))
        ixps = ixps.dimshuffle(1, 0, 2)               # shape=(seq_length, batch_size, n_in)

        uiq_ps = Unique(False, False, False)(pidxs)  # 再去重
        uiq_ix = self.lt[uiq_ps]

        # 输出端：h*w 得到score
        yxps, yxqs = self.vyx[pidxs], self.vyx[qidxs]
        yxps, yxqs = yxps.dimshuffle(1, 0, 2), yxqs.dimshuffle(1, 0, 2)

        pqs = T.concatenate((pidxs, qidxs))         # 先拼接
        uiq_pqs = Unique(False, False, False)(pqs)  # 再去重
        uiq_yx = self.vyx[uiq_pqs]

        """
        输入t时刻正负样本、t-1时刻隐层，计算当前隐层、当前损失. 公式里省略了时刻t
        # 根据性质：T.dot((m, n), (n, ))得到shape=(m, )，且是矩阵每行与(n, )相乘
            # GRU
            z = sigmoid(ux_z * xp + wh_z * h_pre1)
            r = sigmoid(ux_r * xp + wh_r * h_pre1)
            c = tanh(ux_c * xp + wh_c * (r 点乘 h_pre1))
            h = z * h_pre1 + (1.0 - z) * c
        # 根据性质：T.dot((n, ), (n, ))得到scalar
            upq  = h_pre1 * (xp - xq)
            loss = log(1.0 + e^(-upq))
        """
        def recurrence(ixp_t, yxp_t, yxq_t,
                       mask_t, cx_t_pre1, hx_t_pre1):
            # 特征、隐层都处理成shape=(batch_size, n_hidden)=(n, 20)
            gatesx = T.dot(uix, ixp_t.T) + T.dot(whx, hx_t_pre1.T) + bix  # shape=(4, 20, n)
            ix, fx, gx, ox = sigmoid(gatesx[0]).T, sigmoid(gatesx[1]).T, tanh(gatesx[2]).T, sigmoid(gatesx[3]).T
            cx_t = fx * cx_t_pre1 + ix * gx                         # shape=(n, 20)
            hx_t = ox * tanh(cx_t)                                  # shape=(n, 20)
            # 偏好误差
            upq_t = T.sum(hx_t_pre1 * (yxp_t - yxq_t), axis=1)      # shape=(n, )
            loss_t = T.log(sigmoid(upq_t))                          # shape=(n, )
            loss_t *= mask_t                            # 只在损失这里乘一下0/1向量就可以了
            return [cx_t, hx_t, loss_t]                             # shape=(n, 20), (n, )
        [cx, hx, loss], _ = theano.scan(
            fn=recurrence,
            sequences=[ixps, yxps, yxqs, mask],
            outputs_info=[c0x, h0x, None],
            n_steps=seq_length)     # 保证只循环到最长有效位

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, l2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        seq_l2_sq = (
            T.sum([T.sum(par ** 2) for par in [uix, whx, yxps, yxqs, ixps]]) +
            T.sum([T.sum(par ** 2) for par in [bix]]) / actual_batch_size)
        upq = T.sum(loss)
        seq_costs = (
            - upq / actual_batch_size +
            0.5 * l2 * seq_l2_sq)
        seq_grads = T.grad(seq_costs, self.paramsx)
        seq_updates = [(par, par - lr * gra) for par, gra in zip(self.paramsx, seq_grads)]
        update_ix = T.set_subtensor(uiq_ix, uiq_ix - lr * T.grad(seq_costs, self.lt)[uiq_ps])
        update_yx = T.set_subtensor(uiq_yx, uiq_yx - lr * T.grad(seq_costs, self.vyx)[uiq_pqs])
        seq_updates.append((self.lt, update_ix))
        seq_updates.append((self.vyx, update_yx))   # 会直接更改到seq_updates里
        # ----------------------------------------------------------------------------

        # 输入正、负样本序列及其它参数后，更新变量，返回损失。
        # givens给数据
        start_end = T.ivector()
        self.seq_trainx = theano.function(
            inputs=[start_end],
            outputs=-upq,
            updates=seq_updates,
            givens={
                pidxs: self.tra_buys_masks[start_end],       # 类型是 TensorType(int32, matrix)
                qidxs: self.tra_buys_neg_masks[start_end],   # T.ivector()类型是 TensorType(int32, vector)
                tra_mask: self.tra_masks[start_end]})

    def __theano_predictx__(self, n_in, n_hidden):
        """
        测试阶段再跑一遍训练序列得到各个隐层。用全部数据一次性得出所有用户的表达
        """
        uix, whx = self.uix, self.whx

        tra_mask = T.imatrix()          # shape=(n, 157)
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))     # 获取mini-batch里各序列的长度最大值作为seq_length

        c0x = T.alloc(self.c0x, actual_batch_size, n_hidden)      # shape=(n, 20)
        h0x = T.alloc(self.h0x, actual_batch_size, n_hidden)      # shape=(n, 20)
        bix = T.alloc(self.bix, actual_batch_size, 4, n_hidden)   # shape=(n, 3, 20), n_hidden放在最后
        bix = bix.dimshuffle(1, 2, 0)                             # shape=(3, 20, n)

        pidxs = T.imatrix()
        ixps = self.lt[pidxs]       # shape((actual_batch_size, seq_length, n_in))
        ixps = ixps.dimshuffle(1, 0, 2)               # shape=(seq_length, batch_size, n_in)

        def recurrence(ixp_t, cx_t_pre1, hx_t_pre1):
            # 特征、隐层都处理成shape=(batch_size, n_hidden)=(n, 20)
            gatesx = T.dot(uix, ixp_t.T) + T.dot(whx, hx_t_pre1.T) + bix  # shape=(4, 20, n)
            ix, fx, gx, ox = sigmoid(gatesx[0]).T, sigmoid(gatesx[1]).T, tanh(gatesx[2]).T, sigmoid(gatesx[3]).T
            cx_t = fx * cx_t_pre1 + ix * gx                         # shape=(n, 20)
            hx_t = ox * tanh(cx_t)                                  # shape=(n, 20)
            return [cx_t, hx_t]
        [cx, hx], _ = theano.scan(            # h.shape=(157, n, 20)
            fn=recurrence,
            sequences=ixps,
            outputs_info=[c0x, h0x],
            n_steps=seq_length)
        h = hx

        # 得到batch_hus.shape=(n, 60)，就是这个batch里每个用户的表达hu。
        # 必须要用T.sum()，不然无法建模到theano的graph里、报length not known的错
        hs = h.dimshuffle(1, 0, 2)                      # shape=(batch_size, seq_length, n_hidden)
        hts = hs[                                       # shape=(n, n_hidden)
            T.arange(actual_batch_size),                # 行. 花式索引a[[1,2,3],[2,5,6]]，需给定行列的表示
            T.sum(tra_mask, axis=1) - 1]                # 列。需要mask是'int32'型的

        # givens给数据
        start_end = T.ivector()
        self.seq_predictx = theano.function(
            inputs=[start_end],
            outputs=hts,
            givens={
                pidxs: self.tra_buys_masks[start_end],       # 类型是 TensorType(int32, matrix)
                tra_mask: self.tra_masks[start_end]})

    # PGruParallelRes，为了和其统一进一个代码，多了两个参数。
    def train(self, idxs, epoch_n, epoch_i):
        return self.seq_trainx(idxs)

    def pgru_predict(self, idxs, epoch_n, epoch_i):
        return self.seq_predictx(idxs)

    def pgru_update_trained_items(self, epoch_n, epoch_i):
        # 更新最终的items表达
        items = self.vyx.get_value(borrow=True)    # self.lt是shared，用get_value()。
        self.trained_items.set_value(np.asarray(items, dtype=theano.config.floatX), borrow=True)


# 输出时：h*v → sigmoid(T.sum(hx*(vxp-vxq), axis=1) + T.sum(hi*(vip-viq), axis=1) + T.sum(ht*(vtp-vtq), axis=1))
# 预测时：h*v → np.dot([hx, hi, ht], [ix, ii, it].T)
# ======================================================================================================================
class PLstmParallelRes(LstmBasic):
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_in, n_hidden,
                 n_img, n_txt, fea_img, fea_txt):
        super(PLstmParallelRes, self).__init__(train, test, alpha_lambda, n_user, n_item, n_in, n_hidden)
        # 图文特征
        self.ft = theano.shared(borrow=True, value=np.asarray(fea_txt, dtype=theano.config.floatX))  # shape=(n, 100)
        self.fi = theano.shared(borrow=True, value=np.asarray(fea_img, dtype=theano.config.floatX))  # shape=(n, 1024)
        rang = 0.5
        # input
        lt = uniform(-rang, rang, (n_item + 1, n_in))   # 多出来一个(填充符)，存放用于补齐用户购买序列/实际不存在的item
        self.lt = theano.shared(borrow=True, value=lt.astype(theano.config.floatX))
        # hidden state
        uix = uniform(-rang, rang, (4, n_hidden, n_in))
        uit = uniform(-rang, rang, (4, n_hidden, n_txt))
        uii = uniform(-rang, rang, (4, n_hidden, n_img))    # 因为他文章里没对图文特征做降维处理
        whx = uniform(-rang, rang, (4, n_hidden, n_hidden))
        wht = uniform(-rang, rang, (4, n_hidden, n_hidden))
        whi = uniform(-rang, rang, (4, n_hidden, n_hidden))
        c0x = np.zeros((n_hidden, ), dtype=theano.config.floatX)
        c0t = np.zeros((n_hidden, ), dtype=theano.config.floatX)
        c0i = np.zeros((n_hidden, ), dtype=theano.config.floatX)
        h0x = np.zeros((n_hidden, ), dtype=theano.config.floatX)
        h0t = np.zeros((n_hidden, ), dtype=theano.config.floatX)
        h0i = np.zeros((n_hidden, ), dtype=theano.config.floatX)
        bix = np.zeros((4, n_hidden), dtype=theano.config.floatX)
        bit = np.zeros((4, n_hidden), dtype=theano.config.floatX)
        bii = np.zeros((4, n_hidden), dtype=theano.config.floatX)
        self.uix = theano.shared(borrow=True, value=uix.astype(theano.config.floatX))
        self.uit = theano.shared(borrow=True, value=uit.astype(theano.config.floatX))
        self.uii = theano.shared(borrow=True, value=uii.astype(theano.config.floatX))
        self.whx = theano.shared(borrow=True, value=whx.astype(theano.config.floatX))
        self.wht = theano.shared(borrow=True, value=wht.astype(theano.config.floatX))
        self.whi = theano.shared(borrow=True, value=whi.astype(theano.config.floatX))
        self.h0x = theano.shared(borrow=True, value=c0x)
        self.c0t = theano.shared(borrow=True, value=c0t)
        self.c0i = theano.shared(borrow=True, value=c0i)
        self.c0x = theano.shared(borrow=True, value=h0x)
        self.h0t = theano.shared(borrow=True, value=h0t)
        self.h0i = theano.shared(borrow=True, value=h0i)
        self.bix = theano.shared(borrow=True, value=bix)
        self.bit = theano.shared(borrow=True, value=bit)
        self.bii = theano.shared(borrow=True, value=bii)
        # output
        vyx = uniform(-rang, rang, (n_item + 1, n_hidden))       # 和lt一样。
        vyt = uniform(-rang, rang, (n_item + 1, n_hidden))
        vyi = uniform(-rang, rang, (n_item + 1, n_hidden))
        self.vyx = theano.shared(borrow=True, value=vyx.astype(theano.config.floatX))
        self.vyt = theano.shared(borrow=True, value=vyt.astype(theano.config.floatX))
        self.vyi = theano.shared(borrow=True, value=vyi.astype(theano.config.floatX))
        # square loss
        self.paramsx = [self.uix, self.whx, self.bix]       # self.lt单独进行更新。
        self.paramst = [self.uit, self.wht, self.bit]
        self.paramsi = [self.uii, self.whi, self.bii]
        self.params = [
            self.uix, self.whx, self.bix,
            self.uit, self.wht, self.bit,
            self.uii, self.whi, self.bii]
        self.l2_sqrx = (
            T.sum(self.lt ** 2) + T.sum(self.vyx ** 2) +
            T.sum([T.sum(param ** 2) for param in self.paramsx]))
        self.l2_sqrt = (
            T.sum(self.vyt ** 2) +
            T.sum([T.sum(param ** 2) for param in self.paramst]))
        self.l2_sqri = (
            T.sum(self.vyi ** 2) +
            T.sum([T.sum(param ** 2) for param in self.paramsi]))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * (self.l2_sqrx + self.l2_sqrt + self.l2_sqri))
        self.__theano_trainx__(n_in, n_hidden)
        self.__theano_traint__(n_in, n_hidden)
        self.__theano_traini__(n_in, n_hidden)
        self.__theano_predictx__(n_in, n_hidden)
        self.__theano_predictt__(n_in, n_hidden)
        self.__theano_predicti__(n_in, n_hidden)

    def __theano_trainx__(self, n_in, n_hidden):
        """
        训练阶段跑一遍训练序列
        """
        # self.alpha_lambda = ['alpha', 'lambda', 'fea_random_zero']
        uix, whx = self.uix, self.whx

        tra_mask = T.imatrix()                          # shape=(n, 157)
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))     # 获取mini-batch里各序列的长度最大值作为seq_length
        mask = tra_mask.T                               # shape=(157, n)

        c0x = T.alloc(self.c0x, actual_batch_size, n_hidden)
        h0x = T.alloc(self.h0x, actual_batch_size, n_hidden)      # shape=(n, 40)
        bix = T.alloc(self.bix, actual_batch_size, 4, n_hidden)   # shape=(n, 3, 40), n_hidden放在最后
        bix = bix.dimshuffle(1, 2, 0)                             # shape=(3, 40, n)

        # 输入端：只输入购买的商品即可。
        pidxs, qidxs = T.imatrix(), T.imatrix()     # TensorType(int32, matrix)
        ixps = self.lt[pidxs]       # shape((actual_batch_size, seq_length, n_in))
        ixps = ixps.dimshuffle(1, 0, 2)               # shape=(seq_length, batch_size, n_in)

        uiq_ps = Unique(False, False, False)(pidxs)  # 再去重
        uiq_ix = self.lt[uiq_ps]

        # 输出端：h*w 得到score
        yxps, yxqs = self.vyx[pidxs], self.vyx[qidxs]
        yxps, yxqs = yxps.dimshuffle(1, 0, 2), yxqs.dimshuffle(1, 0, 2)

        pqs = T.concatenate((pidxs, qidxs))         # 先拼接
        uiq_pqs = Unique(False, False, False)(pqs)  # 再去重
        uiq_yx = self.vyx[uiq_pqs]

        """
        输入t时刻正负样本、t-1时刻隐层，计算当前隐层、当前损失. 公式里省略了时刻t
        # 根据性质：T.dot((m, n), (n, ))得到shape=(m, )，且是矩阵每行与(n, )相乘
            # GRU
            z = sigmoid(ux_z * xp + wh_z * h_pre1)
            r = sigmoid(ux_r * xp + wh_r * h_pre1)
            c = tanh(ux_c * xp + wh_c * (r 点乘 h_pre1))
            h = z * h_pre1 + (1.0 - z) * c
        # 根据性质：T.dot((n, ), (n, ))得到scalar
            upq  = h_pre1 * (xp - xq)
            loss = log(1.0 + e^(-upq))
        """
        def recurrence(ixp_t, yxp_t, yxq_t,
                       mask_t, cx_t_pre1, hx_t_pre1):
            # 特征、隐层都处理成shape=(batch_size, n_hidden)=(n, 20)
            gatesx = T.dot(uix, ixp_t.T) + T.dot(whx, hx_t_pre1.T) + bix  # shape=(4, 20, n)
            ix, fx, gx, ox = sigmoid(gatesx[0]).T, sigmoid(gatesx[1]).T, tanh(gatesx[2]).T, sigmoid(gatesx[3]).T
            cx_t = fx * cx_t_pre1 + ix * gx
            hx_t = ox * tanh(cx_t)             # shape=(n, 20)
            # 偏好误差
            upq_t = T.sum(hx_t_pre1 * (yxp_t - yxq_t), axis=1)     # shape=(n, )
            loss_t = T.log(sigmoid(upq_t))                      # shape=(n, )
            loss_t *= mask_t                                    # 只在损失这里乘一下0/1向量就可以了
            return [cx_t, hx_t, loss_t]                         # shape=(n, 20), (n, )
        [cx, hx, loss], _ = theano.scan(
            fn=recurrence,
            sequences=[ixps, yxps, yxqs, mask],
            outputs_info=[c0x, h0x, None],
            n_steps=seq_length)     # 保证只循环到最长有效位

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, l2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        seq_l2_sq = (
            T.sum([T.sum(par ** 2) for par in [uix, whx, yxps, yxqs, ixps]]) +
            T.sum([T.sum(par ** 2) for par in [bix]]) / actual_batch_size)
        upq = T.sum(loss)
        seq_costs = (
            - upq / actual_batch_size +
            0.5 * l2 * seq_l2_sq)
        seq_grads = T.grad(seq_costs, self.paramsx)
        seq_updates = [(par, par - lr * gra) for par, gra in zip(self.paramsx, seq_grads)]
        update_ix = T.set_subtensor(uiq_ix, uiq_ix - lr * T.grad(seq_costs, self.lt)[uiq_ps])
        update_yx = T.set_subtensor(uiq_yx, uiq_yx - lr * T.grad(seq_costs, self.vyx)[uiq_pqs])
        seq_updates.append((self.lt, update_ix))
        seq_updates.append((self.vyx, update_yx))   # 会直接更改到seq_updates里
        # ----------------------------------------------------------------------------

        # 输入正、负样本序列及其它参数后，更新变量，返回损失。
        # givens给数据
        start_end = T.ivector()
        self.seq_trainx = theano.function(
            inputs=[start_end],
            outputs=-upq,
            updates=seq_updates,
            givens={
                pidxs: self.tra_buys_masks[start_end],       # 类型是 TensorType(int32, matrix)
                qidxs: self.tra_buys_neg_masks[start_end],   # T.ivector()类型是 TensorType(int32, vector)
                tra_mask: self.tra_masks[start_end]})

    def __theano_traint__(self, n_in, n_hidden):
        """
        训练阶段跑一遍训练序列
        """
        # self.alpha_lambda = ['alpha', 'lambda', 'fea_random_zero']
        uix, whx = self.uix, self.whx
        uit, wht = self.uit, self.wht

        tra_mask = T.imatrix()                          # shape=(n, 157)
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))     # 获取mini-batch里各序列的长度最大值作为seq_length
        mask = tra_mask.T                               # shape=(157, n)

        c0x = T.alloc(self.c0x, actual_batch_size, n_hidden)
        c0t = T.alloc(self.c0t, actual_batch_size, n_hidden)
        h0x = T.alloc(self.h0x, actual_batch_size, n_hidden)      # shape=(n, 40)
        h0t = T.alloc(self.h0t, actual_batch_size, n_hidden)
        bix = T.alloc(self.bix, actual_batch_size, 4, n_hidden)   # shape=(n, 3, 40), n_hidden放在最后
        bit = T.alloc(self.bit, actual_batch_size, 4, n_hidden)
        bix = bix.dimshuffle(1, 2, 0)                             # shape=(3, 40, n)
        bit = bit.dimshuffle(1, 2, 0)

        # 输入端：只输入购买的商品即可。
        pidxs, qidxs = T.imatrix(), T.imatrix()     # TensorType(int32, matrix)
        ixps = self.lt[pidxs]       # shape((actual_batch_size, seq_length, n_in))
        itps = self.ft[pidxs]       # shape((actual_batch_size, seq_length, n_txt))
        ixps = ixps.dimshuffle(1, 0, 2)               # shape=(seq_length, batch_size, n_in)
        itps = itps.dimshuffle(1, 0, 2)

        # 输出端：h*w 得到score
        yxps, yxqs = self.vyx[pidxs], self.vyx[qidxs]
        ytps, ytqs = self.vyt[pidxs], self.vyt[qidxs]
        yxps, yxqs = yxps.dimshuffle(1, 0, 2), yxqs.dimshuffle(1, 0, 2)
        ytps, ytqs = ytps.dimshuffle(1, 0, 2), ytqs.dimshuffle(1, 0, 2)

        pqs = T.concatenate((pidxs, qidxs))         # 先拼接
        uiq_pqs = Unique(False, False, False)(pqs)  # 再去重
        uiq_yt = self.vyt[uiq_pqs]

        """
        输入t时刻正负样本、t-1时刻隐层，计算当前隐层、当前损失. 公式里省略了时刻t
        # 根据性质：T.dot((m, n), (n, ))得到shape=(m, )，且是矩阵每行与(n, )相乘
            # GRU
            z = sigmoid(ux_z * xp + wh_z * h_pre1)
            r = sigmoid(ux_r * xp + wh_r * h_pre1)
            c = tanh(ux_c * xp + wh_c * (r 点乘 h_pre1))
            h = z * h_pre1 + (1.0 - z) * c
        # 根据性质：T.dot((n, ), (n, ))得到scalar
            upq  = h_pre1 * (xp - xq)
            loss = log(1.0 + e^(-upq))
        """
        def recurrence(ixp_t, yxp_t, yxq_t,
                       itp_t, ytp_t, ytq_t,
                       mask_t, cx_t_pre1, ct_t_pre1, hx_t_pre1, ht_t_pre1):
            # 特征、隐层都处理成shape=(batch_size, n_hidden)=(n, 20)
            gatesx = T.dot(uix, ixp_t.T) + T.dot(whx, hx_t_pre1.T) + bix  # shape=(4, 20, n)
            gatest = T.dot(uit, itp_t.T) + T.dot(wht, ht_t_pre1.T) + bit  # shape=(4, 20, n)
            ix, fx, gx, ox = sigmoid(gatesx[0]).T, sigmoid(gatesx[1]).T, tanh(gatesx[2]).T, sigmoid(gatesx[3]).T
            it, ft, gt, ot = sigmoid(gatest[0]).T, sigmoid(gatest[1]).T, tanh(gatest[2]).T, sigmoid(gatest[3]).T
            cx_t = fx * cx_t_pre1 + ix * gx      # shape=(n, 20)
            ct_t = ft * ct_t_pre1 + it * gt      # shape=(n, 20)
            hx_t = ox * tanh(cx_t)             # shape=(n, 20)
            ht_t = ot * tanh(ct_t)             # shape=(n, 20)
            # 偏好误差
            upq_t = (
                T.sum(hx_t_pre1 * (yxp_t - yxq_t), axis=1) +
                T.sum(ht_t_pre1 * (ytp_t - ytq_t), axis=1))     # shape=(n, )
            loss_t = T.log(sigmoid(upq_t))                      # shape=(n, )
            loss_t *= mask_t                                    # 只在损失这里乘一下0/1向量就可以了
            return [cx_t, ct_t, hx_t, ht_t, loss_t]                         # shape=(n, 20), (n, )
        [cx, ct, hx, ht, loss], _ = theano.scan(
            fn=recurrence,
            sequences=[ixps, yxps, yxqs,
                       itps, ytps, ytqs, mask],
            outputs_info=[c0x, c0t, h0x, h0t, None],
            n_steps=seq_length)     # 保证只循环到最长有效位

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, l2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        seq_l2_sq = (
            T.sum([T.sum(par ** 2) for par in [uix, whx, yxps, yxqs, ixps,
                                               uit, wht, ytps, ytqs]]) +
            T.sum([T.sum(par ** 2) for par in [bix, bit]]) / actual_batch_size)
        upq = T.sum(loss)
        seq_costs = (
            - upq / actual_batch_size +
            0.5 * l2 * seq_l2_sq)
        seq_grads = T.grad(seq_costs, self.paramst)
        seq_updates = [(par, par - lr * gra) for par, gra in zip(self.paramst, seq_grads)]
        update_yt = T.set_subtensor(uiq_yt, uiq_yt - lr * T.grad(seq_costs, self.vyt)[uiq_pqs])
        seq_updates.append((self.vyt, update_yt))   # 会直接更改到seq_updates里
        # ----------------------------------------------------------------------------

        # 输入正、负样本序列及其它参数后，更新变量，返回损失。
        # givens给数据
        start_end = T.ivector()
        self.seq_traint = theano.function(
            inputs=[start_end],
            outputs=-upq,
            updates=seq_updates,
            givens={
                pidxs: self.tra_buys_masks[start_end],       # 类型是 TensorType(int32, matrix)
                qidxs: self.tra_buys_neg_masks[start_end],   # T.ivector()类型是 TensorType(int32, vector)
                tra_mask: self.tra_masks[start_end]})

    def __theano_traini__(self, n_in, n_hidden):
        """
        训练阶段跑一遍训练序列
        """
        # self.alpha_lambda = ['alpha', 'lambda', 'fea_random_zero']
        uix, whx = self.uix, self.whx
        uit, wht = self.uit, self.wht
        uii, whi = self.uii, self.whi

        tra_mask = T.imatrix()                          # shape=(n, 157)
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))     # 获取mini-batch里各序列的长度最大值作为seq_length
        mask = tra_mask.T                               # shape=(157, n)

        c0x = T.alloc(self.c0x, actual_batch_size, n_hidden)
        c0t = T.alloc(self.c0t, actual_batch_size, n_hidden)
        c0i = T.alloc(self.c0i, actual_batch_size, n_hidden)
        h0x = T.alloc(self.h0x, actual_batch_size, n_hidden)      # shape=(n, 40)
        h0t = T.alloc(self.h0t, actual_batch_size, n_hidden)
        h0i = T.alloc(self.h0i, actual_batch_size, n_hidden)
        bix = T.alloc(self.bix, actual_batch_size, 4, n_hidden)   # shape=(n, 3, 40), n_hidden放在最后
        bit = T.alloc(self.bit, actual_batch_size, 4, n_hidden)
        bii = T.alloc(self.bii, actual_batch_size, 4, n_hidden)
        bix = bix.dimshuffle(1, 2, 0)                             # shape=(3, 40, n)
        bit = bit.dimshuffle(1, 2, 0)
        bii = bii.dimshuffle(1, 2, 0)

        # 输入端：只输入购买的商品即可。
        pidxs, qidxs = T.imatrix(), T.imatrix()     # TensorType(int32, matrix)
        ixps = self.lt[pidxs]       # shape((actual_batch_size, seq_length, n_in))
        itps = self.ft[pidxs]       # shape((actual_batch_size, seq_length, n_txt))
        iips = self.fi[pidxs]       # shape((actual_batch_size, seq_length, n_img))
        ixps = ixps.dimshuffle(1, 0, 2)               # shape=(seq_length, batch_size, n_in)
        itps = itps.dimshuffle(1, 0, 2)
        iips = iips.dimshuffle(1, 0, 2)

        # 输出端：h*w 得到score
        yxps, yxqs = self.vyx[pidxs], self.vyx[qidxs]
        ytps, ytqs = self.vyt[pidxs], self.vyt[qidxs]
        yips, yiqs = self.vyi[pidxs], self.vyi[qidxs]
        yxps, yxqs = yxps.dimshuffle(1, 0, 2), yxqs.dimshuffle(1, 0, 2)
        ytps, ytqs = ytps.dimshuffle(1, 0, 2), ytqs.dimshuffle(1, 0, 2)
        yips, yiqs = yips.dimshuffle(1, 0, 2), yiqs.dimshuffle(1, 0, 2)

        pqs = T.concatenate((pidxs, qidxs))         # 先拼接
        uiq_pqs = Unique(False, False, False)(pqs)  # 再去重
        uiq_yi = self.vyi[uiq_pqs]

        """
        输入t时刻正负样本、t-1时刻隐层，计算当前隐层、当前损失. 公式里省略了时刻t
        # 根据性质：T.dot((m, n), (n, ))得到shape=(m, )，且是矩阵每行与(n, )相乘
            # GRU
            z = sigmoid(ux_z * xp + wh_z * h_pre1)
            r = sigmoid(ux_r * xp + wh_r * h_pre1)
            c = tanh(ux_c * xp + wh_c * (r 点乘 h_pre1))
            h = z * h_pre1 + (1.0 - z) * c
        # 根据性质：T.dot((n, ), (n, ))得到scalar
            upq  = h_pre1 * (xp - xq)
            loss = log(1.0 + e^(-upq))
        """
        def recurrence(ixp_t, yxp_t, yxq_t,
                       itp_t, ytp_t, ytq_t,
                       iip_t, yip_t, yiq_t,
                       mask_t, cx_t_pre1, ct_t_pre1, ci_t_pre1, hx_t_pre1, ht_t_pre1, hi_t_pre1):
            # 特征、隐层都处理成shape=(batch_size, n_hidden)=(n, 20)
            gatesx = T.dot(uix, ixp_t.T) + T.dot(whx, hx_t_pre1.T) + bix  # shape=(4, 20, n)
            gatest = T.dot(uit, itp_t.T) + T.dot(wht, ht_t_pre1.T) + bit  # shape=(4, 20, n)
            gatesi = T.dot(uii, iip_t.T) + T.dot(whi, hi_t_pre1.T) + bii  # shape=(4, 20, n)
            ix, fx, gx, ox = sigmoid(gatesx[0]).T, sigmoid(gatesx[1]).T, tanh(gatesx[2]).T, sigmoid(gatesx[3]).T
            it, ft, gt, ot = sigmoid(gatest[0]).T, sigmoid(gatest[1]).T, tanh(gatest[2]).T, sigmoid(gatest[3]).T
            ii, fi, gi, oi = sigmoid(gatesi[0]).T, sigmoid(gatesi[1]).T, tanh(gatesi[2]).T, sigmoid(gatesi[3]).T
            cx_t = fx * cx_t_pre1 + ix * gx      # shape=(n, 20)
            ct_t = ft * ct_t_pre1 + it * gt      # shape=(n, 20)
            ci_t = fi * ci_t_pre1 + ii * gi      # shape=(n, 20)
            hx_t = ox * tanh(cx_t)             # shape=(n, 20)
            ht_t = ot * tanh(ct_t)             # shape=(n, 20)
            hi_t = oi * tanh(ci_t)             # shape=(n, 20)
            # 偏好误差
            upq_t = (
                T.sum(hx_t_pre1 * (yxp_t - yxq_t), axis=1) +
                T.sum(ht_t_pre1 * (ytp_t - ytq_t), axis=1) +
                T.sum(hi_t_pre1 * (yip_t - yiq_t), axis=1))     # shape=(n, )
            loss_t = T.log(sigmoid(upq_t))                      # shape=(n, )
            loss_t *= mask_t                                    # 只在损失这里乘一下0/1向量就可以了
            return [cx_t, ct_t, ci_t, hx_t, ht_t, hi_t, loss_t]                         # shape=(n, 20), (n, )
        [cx, ct, ci, hx, ht, hi, loss], _ = theano.scan(
            fn=recurrence,
            sequences=[ixps, yxps, yxqs,
                       itps, ytps, ytqs,
                       iips, yips, yiqs, mask],
            outputs_info=[c0x, c0t, c0i, h0x, h0t, h0i, None],
            n_steps=seq_length)     # 保证只循环到最长有效位

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, l2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        seq_l2_sq = (
            T.sum([T.sum(par ** 2) for par in [uix, whx, yxps, yxqs, ixps,
                                               uit, wht, ytps, ytqs,
                                               uii, whi, yips, yiqs]]) +
            T.sum([T.sum(par ** 2) for par in [bix, bit, bii]]) / actual_batch_size)
        upq = T.sum(loss)
        seq_costs = (
            - upq / actual_batch_size +
            0.5 * l2 * seq_l2_sq)
        seq_grads = T.grad(seq_costs, self.paramsi)
        seq_updates = [(par, par - lr * gra) for par, gra in zip(self.paramsi, seq_grads)]
        update_yi = T.set_subtensor(uiq_yi, uiq_yi - lr * T.grad(seq_costs, self.vyi)[uiq_pqs])
        seq_updates.append((self.vyi, update_yi))   # 会直接更改到seq_updates里
        # ----------------------------------------------------------------------------

        # 输入正、负样本序列及其它参数后，更新变量，返回损失。
        # givens给数据
        start_end = T.ivector()
        self.seq_traini = theano.function(
            inputs=[start_end],
            outputs=-upq,
            updates=seq_updates,
            givens={
                pidxs: self.tra_buys_masks[start_end],       # 类型是 TensorType(int32, matrix)
                qidxs: self.tra_buys_neg_masks[start_end],   # T.ivector()类型是 TensorType(int32, vector)
                tra_mask: self.tra_masks[start_end]})

    def __theano_predictx__(self, n_in, n_hidden):
        """
        测试阶段再跑一遍训练序列得到各个隐层。用全部数据一次性得出所有用户的表达
        """
        uix, whx = self.uix, self.whx

        tra_mask = T.imatrix()          # shape=(n, 157)
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))     # 获取mini-batch里各序列的长度最大值作为seq_length

        c0x = T.alloc(self.c0x, actual_batch_size, n_hidden)
        h0x = T.alloc(self.h0x, actual_batch_size, n_hidden)      # shape=(n, 20)
        bix = T.alloc(self.bix, actual_batch_size, 4, n_hidden)   # shape=(n, 3, 20), n_hidden放在最后
        bix = bix.dimshuffle(1, 2, 0)                             # shape=(3, 20, n)

        pidxs = T.imatrix()
        ixps = self.lt[pidxs]       # shape((actual_batch_size, seq_length, n_in))
        ixps = ixps.dimshuffle(1, 0, 2)               # shape=(seq_length, batch_size, n_in)

        def recurrence(ixp_t, cx_t_pre1, hx_t_pre1,):
            # 特征、隐层都处理成shape=(batch_size, n_hidden)=(n, 20)
            # 隐层计算
            gatesx = T.dot(uix, ixp_t.T) + T.dot(whx, hx_t_pre1.T) + bix  # shape=(4, 20, n)
            ix, fx, gx, ox = sigmoid(gatesx[0]).T, sigmoid(gatesx[1]).T, tanh(gatesx[2]).T, sigmoid(gatesx[3]).T
            cx_t = fx * cx_t_pre1 + ix * gx      # shape=(n, 20)
            hx_t = ox * tanh(cx_t)             # shape=(n, 20)
            return [cx_t, hx_t]
        [cx, hx], _ = theano.scan(              # h.shape=(157, n, 20)
            fn=recurrence,
            sequences=ixps,
            outputs_info=[c0x, h0x],
            n_steps=seq_length)
        h = hx

        # 得到batch_hus.shape=(n, 60)，就是这个batch里每个用户的表达hu。
        # 必须要用T.sum()，不然无法建模到theano的graph里、报length not known的错
        hs = h.dimshuffle(1, 0, 2)                      # shape=(batch_size, seq_length, n_hidden)
        hts = hs[                                       # shape=(n, n_hidden)
            T.arange(actual_batch_size),                # 行. 花式索引a[[1,2,3],[2,5,6]]，需给定行列的表示
            T.sum(tra_mask, axis=1) - 1]                # 列。需要mask是'int32'型的

        # givens给数据
        start_end = T.ivector()
        self.seq_predictx = theano.function(
            inputs=[start_end],
            outputs=hts,
            givens={
                pidxs: self.tra_buys_masks[start_end],       # 类型是 TensorType(int32, matrix)
                tra_mask: self.tra_masks[start_end]})

    def __theano_predictt__(self, n_in, n_hidden):
        """
        测试阶段再跑一遍训练序列得到各个隐层。用全部数据一次性得出所有用户的表达
        """
        uix, whx = self.uix, self.whx
        uit, wht = self.uit, self.wht

        tra_mask = T.imatrix()          # shape=(n, 157)
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))     # 获取mini-batch里各序列的长度最大值作为seq_length

        c0x = T.alloc(self.c0x, actual_batch_size, n_hidden)
        c0t = T.alloc(self.c0t, actual_batch_size, n_hidden)
        h0x = T.alloc(self.h0x, actual_batch_size, n_hidden)      # shape=(n, 20)
        h0t = T.alloc(self.h0t, actual_batch_size, n_hidden)
        bix = T.alloc(self.bix, actual_batch_size, 4, n_hidden)   # shape=(n, 3, 20), n_hidden放在最后
        bit = T.alloc(self.bit, actual_batch_size, 4, n_hidden)
        bix = bix.dimshuffle(1, 2, 0)                             # shape=(3, 20, n)
        bit = bit.dimshuffle(1, 2, 0)

        pidxs = T.imatrix()
        ixps = self.lt[pidxs]       # shape((actual_batch_size, seq_length, n_in))
        itps = self.ft[pidxs]       # shape((actual_batch_size, seq_length, n_txt))
        ixps = ixps.dimshuffle(1, 0, 2)               # shape=(seq_length, batch_size, n_in)
        itps = itps.dimshuffle(1, 0, 2)

        def recurrence(ixp_t, itp_t, cx_t_pre1, ct_t_pre1, hx_t_pre1, ht_t_pre1):
            # 特征、隐层都处理成shape=(batch_size, n_hidden)=(n, 20)
            # 隐层计算
            gatesx = T.dot(uix, ixp_t.T) + T.dot(whx, hx_t_pre1.T) + bix  # shape=(4, 20, n)
            gatest = T.dot(uit, itp_t.T) + T.dot(wht, ht_t_pre1.T) + bit  # shape=(4, 20, n)
            ix, fx, gx, ox = sigmoid(gatesx[0]).T, sigmoid(gatesx[1]).T, tanh(gatesx[2]).T, sigmoid(gatesx[3]).T
            it, ft, gt, ot = sigmoid(gatest[0]).T, sigmoid(gatest[1]).T, tanh(gatest[2]).T, sigmoid(gatest[3]).T
            cx_t = fx * cx_t_pre1 + ix * gx      # shape=(n, 20)
            ct_t = ft * ct_t_pre1 + it * gt      # shape=(n, 20)
            hx_t = ox * tanh(cx_t)             # shape=(n, 20)
            ht_t = ot * tanh(ct_t)             # shape=(n, 20)
            return [cx_t, ct_t, hx_t, ht_t]
        [cx, ct, hx, ht], _ = theano.scan(              # h.shape=(157, n, 20)
            fn=recurrence,
            sequences=[ixps, itps],
            outputs_info=[c0x, c0t, h0x, h0t],
            n_steps=seq_length)
        h = T.concatenate((hx, ht), axis=2)     # h.shape=(157, n, 60)

        # 得到batch_hus.shape=(n, 60)，就是这个batch里每个用户的表达hu。
        # 必须要用T.sum()，不然无法建模到theano的graph里、报length not known的错
        hs = h.dimshuffle(1, 0, 2)                      # shape=(batch_size, seq_length, n_hidden)
        hts = hs[                                       # shape=(n, n_hidden)
            T.arange(actual_batch_size),                # 行. 花式索引a[[1,2,3],[2,5,6]]，需给定行列的表示
            T.sum(tra_mask, axis=1) - 1]                # 列。需要mask是'int32'型的

        # givens给数据
        start_end = T.ivector()
        self.seq_predictt = theano.function(
            inputs=[start_end],
            outputs=hts,
            givens={
                pidxs: self.tra_buys_masks[start_end],       # 类型是 TensorType(int32, matrix)
                tra_mask: self.tra_masks[start_end]})

    def __theano_predicti__(self, n_in, n_hidden):
        """
        测试阶段再跑一遍训练序列得到各个隐层。用全部数据一次性得出所有用户的表达
        """
        uix, whx = self.uix, self.whx
        uit, wht = self.uit, self.wht
        uii, whi = self.uii, self.whi

        tra_mask = T.imatrix()          # shape=(n, 157)
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))     # 获取mini-batch里各序列的长度最大值作为seq_length

        c0x = T.alloc(self.c0x, actual_batch_size, n_hidden)
        c0t = T.alloc(self.c0t, actual_batch_size, n_hidden)
        c0i = T.alloc(self.c0i, actual_batch_size, n_hidden)
        h0x = T.alloc(self.h0x, actual_batch_size, n_hidden)      # shape=(n, 20)
        h0t = T.alloc(self.h0t, actual_batch_size, n_hidden)
        h0i = T.alloc(self.h0i, actual_batch_size, n_hidden)
        bix = T.alloc(self.bix, actual_batch_size, 4, n_hidden)   # shape=(n, 3, 20), n_hidden放在最后
        bit = T.alloc(self.bit, actual_batch_size, 4, n_hidden)
        bii = T.alloc(self.bii, actual_batch_size, 4, n_hidden)
        bix = bix.dimshuffle(1, 2, 0)                             # shape=(3, 20, n)
        bit = bit.dimshuffle(1, 2, 0)
        bii = bii.dimshuffle(1, 2, 0)

        pidxs = T.imatrix()
        ixps = self.lt[pidxs]       # shape((actual_batch_size, seq_length, n_in))
        itps = self.ft[pidxs]       # shape((actual_batch_size, seq_length, n_txt))
        iips = self.fi[pidxs]       # shape((actual_batch_size, seq_length, n_img))
        ixps = ixps.dimshuffle(1, 0, 2)               # shape=(seq_length, batch_size, n_in)
        itps = itps.dimshuffle(1, 0, 2)
        iips = iips.dimshuffle(1, 0, 2)

        def recurrence(ixp_t, itp_t, iip_t, cx_t_pre1, ct_t_pre1, ci_t_pre1, hx_t_pre1, ht_t_pre1, hi_t_pre1):
            # 特征、隐层都处理成shape=(batch_size, n_hidden)=(n, 20)
            # 隐层计算
            gatesx = T.dot(uix, ixp_t.T) + T.dot(whx, hx_t_pre1.T) + bix  # shape=(4, 20, n)
            gatest = T.dot(uit, itp_t.T) + T.dot(wht, ht_t_pre1.T) + bit  # shape=(4, 20, n)
            gatesi = T.dot(uii, iip_t.T) + T.dot(whi, hi_t_pre1.T) + bii  # shape=(4, 20, n)
            ix, fx, gx, ox = sigmoid(gatesx[0]).T, sigmoid(gatesx[1]).T, tanh(gatesx[2]).T, sigmoid(gatesx[3]).T
            it, ft, gt, ot = sigmoid(gatest[0]).T, sigmoid(gatest[1]).T, tanh(gatest[2]).T, sigmoid(gatest[3]).T
            ii, fi, gi, oi = sigmoid(gatesi[0]).T, sigmoid(gatesi[1]).T, tanh(gatesi[2]).T, sigmoid(gatesi[3]).T
            cx_t = fx * cx_t_pre1 + ix * gx      # shape=(n, 20)
            ct_t = ft * ct_t_pre1 + it * gt      # shape=(n, 20)
            ci_t = fi * ci_t_pre1 + ii * gi      # shape=(n, 20)
            hx_t = ox * tanh(cx_t)             # shape=(n, 20)
            ht_t = ot * tanh(ct_t)             # shape=(n, 20)
            hi_t = oi * tanh(ci_t)             # shape=(n, 20)
            return [cx_t, ct_t, ci_t, hx_t, ht_t, hi_t]
        [cx, ct, ci, hx, ht, hi], _ = theano.scan(              # h.shape=(157, n, 20)
            fn=recurrence,
            sequences=[ixps, itps, iips],
            outputs_info=[c0x, c0t, c0i, h0x, h0t, h0i],
            n_steps=seq_length)
        h = T.concatenate((hx, ht, hi), axis=2)     # h.shape=(157, n, 60)

        # 得到batch_hus.shape=(n, 60)，就是这个batch里每个用户的表达hu。
        # 必须要用T.sum()，不然无法建模到theano的graph里、报length not known的错
        hs = h.dimshuffle(1, 0, 2)                      # shape=(batch_size, seq_length, n_hidden)
        hts = hs[                                       # shape=(n, n_hidden)
            T.arange(actual_batch_size),                # 行. 花式索引a[[1,2,3],[2,5,6]]，需给定行列的表示
            T.sum(tra_mask, axis=1) - 1]                # 列。需要mask是'int32'型的

        # givens给数据
        start_end = T.ivector()
        self.seq_predicti = theano.function(
            inputs=[start_end],
            outputs=hts,
            givens={
                pidxs: self.tra_buys_masks[start_end],       # 类型是 TensorType(int32, matrix)
                tra_mask: self.tra_masks[start_end]})

    def train(self, idxs, epoch_n, epoch_i):
        if epoch_i < epoch_n // 3:
            return self.seq_trainx(idxs)
        elif epoch_i < epoch_n * 2 // 3:
            return self.seq_traint(idxs)
        else:
            return self.seq_traini(idxs)

    def pgru_predict(self, idxs, epoch_n, epoch_i):
        if epoch_i < epoch_n // 3:
            return self.seq_predictx(idxs)
        elif epoch_i < epoch_n * 2 // 3:
            return self.seq_predictt(idxs)
        else:
            return self.seq_predicti(idxs)

    def pgru_update_trained_items(self, epoch_n, epoch_i):
        vyx = self.vyx.get_value(borrow=True)    # self.lt是shared，用get_value()。
        if epoch_i < epoch_n // 3:
            self.trained_items.set_value(np.asarray(vyx, dtype=theano.config.floatX), borrow=True)
        elif epoch_i < epoch_n * 2 // 3:
            vyt = self.vyt.get_value(borrow=True)
            items = np.concatenate((vyx, vyt), axis=1)
            self.trained_items.set_value(np.asarray(items, dtype=theano.config.floatX), borrow=True)
        else:
            vyt = self.vyt.get_value(borrow=True)
            vyi = self.vyi.get_value(borrow=True)
            items = np.concatenate((vyx, vyt, vyi), axis=1)
            # 更新最终的items表达
            self.trained_items.set_value(np.asarray(items, dtype=theano.config.floatX), borrow=True)


@exe_time  # 放到待调用函数的定义的上一行
def main():
    print('... construct the class RNN, GRU')


if '__main__' == __name__:
    main()
