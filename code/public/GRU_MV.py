#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:       Qiang Cui:  <cuiqiang1990[at]hotmail.com>
# Descripton:   
# Create Date:  2017-11-14 15:00:00
# Modify Date:  2017-00-00 00:00:00
# Modify Disp:

import datetime
import numpy as np
from numpy.random import uniform
import theano
import theano.tensor as T
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh
from theano.tensor.extra_ops import Unique
from GRU import GruBasic, GruBasic2Units
__docformat__ = 'restructedtext en'


def exe_time(func):
    def new_func(*args, **args2):
        name = func.__name__
        start = datetime.datetime.now()
        print("-- {%s} start: @ %ss" % (name, start.strftime("%Y.%m.%d_%H.%M.%S")))
        back = func(*args, **args2)
        end = datetime.datetime.now()
        print("-- {%s} start: @ %ss" % (name, start.strftime("%Y.%m.%d_%H.%M.%S")))
        print("-- {%s} end:   @ %ss" % (name, end.strftime("%Y.%m.%d_%H.%M.%S")))
        total = (end - start).total_seconds()
        print("-- {%s} total: @ %.2fs = %.2fh" % (name, total, total / 3600.0))
        return back
    return new_func


# Obo的用不着，taobao太大。如果数据集小，可以用。
# ======================================================================================================================
class OboMvGru(GruBasic):
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_in, n_hidden,
                 n_img, n_txt, fea_img, fea_txt, fea_img_ct, fea_txt_ct):
        super(OboMvGru, self).__init__(train, test, alpha_lambda, n_user, n_item, n_in, n_hidden)
        self.fi = theano.shared(borrow=True, value=np.asarray(fea_img, dtype=theano.config.floatX))  # shape=(n, 1024)
        self.ft = theano.shared(borrow=True, value=np.asarray(fea_txt, dtype=theano.config.floatX))  # shape=(n, 100)
        self.fi_ct = theano.shared(borrow=True, value=np.asarray(fea_img_ct, dtype=theano.config.floatX))  # shape=(n, 1024)
        self.ft_ct = theano.shared(borrow=True, value=np.asarray(fea_txt_ct, dtype=theano.config.floatX))  # shape=(n, 100)
        # 其它参数
        rang = 0.5
        lt = uniform(-rang, rang, (n_item + 1, n_in))   # 多出来一个(填充符)，存放用于补齐用户购买序列/实际不存在的item
        mm = uniform(-rang, rang, (n_item + 1, n_in))   # 多模态融合特征，和 lt 一一对应。
        ei = uniform(-rang, rang, (n_in, n_img))                # shape=(20, 1024)
        vt = uniform(-rang, rang, (n_in, n_txt))                # shape=(20, 100)
        self.lt = theano.shared(borrow=True, value=lt.astype(theano.config.floatX))
        self.mm = theano.shared(borrow=True, value=mm.astype(theano.config.floatX))
        self.ei = theano.shared(borrow=True, value=ei.astype(theano.config.floatX))
        self.vt = theano.shared(borrow=True, value=vt.astype(theano.config.floatX))
        self.params = [
            self.ui, self.wh, self.bi,  # self.lt单独进行更新。
            self.ei, self.vt]
        self.l2_sqr = (
            T.sum(self.lt ** 2) +
            T.sum([T.sum(param ** 2) for param in self.params[:3]]))
        self.l2_ev = (
            T.sum([T.sum(param ** 2) for param in self.params[3:]]))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * self.l2_sqr +
            0.5 * self.alpha_lambda[2] * self.l2_ev)
        self.__theano_train__(n_hidden, n_img, n_txt)
        self.__theano_predict__(n_in, n_hidden)

    def __theano_train__(self, n_hidden, n_img, n_txt):
        """
        训练阶段跑一遍训练序列
        """
        # self.alpha_lambda = ['alpha', 'lambda', 'lambda_ev', 'lambda_ae', 'fea_random_zero']
        ui, wh = self.ui, self.wh
        ei, vt = self.ei, self.vt

        tra_mask = T.ivector()
        seq_length = T.sum(tra_mask)                # 有效长度

        h0 = self.h0
        bi = self.bi

        pidxs, qidxs = T.ivector(), T.ivector()
        xps, xqs = self.lt[pidxs], self.lt[qidxs]   # shape((seq_length, n_in))
        ips, iqs = self.fi[pidxs], self.fi[qidxs]   # shape((seq_length, n_img))
        tps, tqs = self.ft[pidxs], self.ft[qidxs]   # shape((seq_length, n_txt))

        pqs = T.concatenate((pidxs, qidxs))         # 先拼接
        uiq_pqs = Unique(False, False, False)(pqs)  # 再去重
        uiq_x = self.lt[uiq_pqs]                    # 相应的items特征

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
        zero = self.alpha_lambda[4]
        if 0.0 == zero:     # 用完全数据
            def recurrence(xp_t, xq_t, ip_t, iq_t, tp_t, tq_t, h_t_pre1):
                # item表达
                mp_t = T.dot(ei, ip_t) + T.dot(vt, tp_t)
                mq_t = T.dot(ei, iq_t) + T.dot(vt, tq_t)
                p_t = T.concatenate((xp_t, mp_t))
                q_t = T.concatenate((xq_t, mq_t))
                # 隐层计算
                z_r = sigmoid(T.dot(ui[:2], p_t) +
                              T.dot(wh[:2], h_t_pre1) + bi[:2])
                z, r = z_r[0], z_r[1]
                c = tanh(T.dot(ui[2], p_t) +
                         T.dot(wh[2], (r * h_t_pre1)) + bi[2])
                h_t = (T.ones_like(z) - z) * h_t_pre1 + z * c
                # 偏好误差
                upq_t = T.dot(h_t_pre1, p_t - q_t)
                loss_t = T.log(sigmoid(upq_t))
                # 重构误差
                loss_ae_t_i = (
                    T.sum((ip_t - T.dot(ei.T, mp_t)) ** 2) +
                    T.sum((iq_t - T.dot(ei.T, mq_t)) ** 2))
                loss_ae_t_t = (
                    T.sum((tp_t - T.dot(vt.T, mp_t)) ** 2) +
                    T.sum((tq_t - T.dot(vt.T, mq_t)) ** 2))
                return [h_t, loss_t, loss_ae_t_i, loss_ae_t_t]
            [h, loss, loss_ae_i, loss_ae_t], _ = theano.scan(
                fn=recurrence,
                sequences=[xps, xqs, ips, iqs, tps, tqs],
                outputs_info=[h0, None, None, None],
                n_steps=seq_length,
                truncate_gradient=-1)
        else:
            # 每条序列训练前都随机whole feature corrupted
            ipsc = self.get_corrupted_input_whole(ips, zero)
            iqsc = self.get_corrupted_input_whole(iqs, zero)
            tpsc = self.get_corrupted_input_whole(tps, zero)
            tqsc = self.get_corrupted_input_whole(tqs, zero)

            def recurrence(xp_t, xq_t, ip_t, iq_t, tp_t, tq_t,
                           ipc_t, iqc_t, tpc_t, tqc_t, h_t_pre1):
                # item表达
                mp_t = T.dot(ei, ipc_t) + T.dot(vt, tpc_t)
                mq_t = T.dot(ei, iqc_t) + T.dot(vt, tqc_t)
                p_t = T.concatenate((xp_t, mp_t))
                q_t = T.concatenate((xq_t, mq_t))
                # 隐层计算
                z_r = sigmoid(T.dot(ui[:2], p_t) +
                              T.dot(wh[:2], h_t_pre1) + bi[:2])
                z, r = z_r[0], z_r[1]
                c = tanh(T.dot(ui[2], p_t) +
                         T.dot(wh[2], (r * h_t_pre1)) + bi[2])
                h_t = (T.ones_like(z) - z) * h_t_pre1 + z * c
                # 偏好误差
                upq_t = T.dot(h_t_pre1, p_t - q_t)
                loss_t = T.log(sigmoid(upq_t))
                # 重构误差
                loss_ae_t_i = (
                    T.sum((ip_t - T.dot(ei.T, mp_t)) ** 2) +
                    T.sum((iq_t - T.dot(ei.T, mq_t)) ** 2))
                loss_ae_t_t = (
                    T.sum((tp_t - T.dot(vt.T, mp_t)) ** 2) +
                    T.sum((tq_t - T.dot(vt.T, mq_t)) ** 2))
                return [h_t, loss_t, loss_ae_t_i, loss_ae_t_t]
            [h, loss, loss_ae_i, loss_ae_t], _ = theano.scan(
                fn=recurrence,
                sequences=[xps, xqs, ips, iqs, tps, tqs, ipsc, iqsc, tpsc, tqsc],
                outputs_info=[h0, None, None, None],
                n_steps=seq_length,
                truncate_gradient=-1)

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, l2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        l2_ev = self.alpha_lambda[2]
        l2_ae = self.alpha_lambda[3]
        seq_l2_sq = T.sum([T.sum(par ** 2) for par in [xps, xqs, ui, wh, bi]])
        seq_l2_ev = T.sum([T.sum(par ** 2) for par in [ei, vt]])
        upq = T.sum(loss)
        ae = (
            0.5 * l2_ae * T.sum(loss_ae_i) / n_img +
            0.5 * l2_ae * T.sum(loss_ae_t) / n_txt)
        seq_costs = (
            - upq + ae +
            0.5 * l2 * seq_l2_sq +
            0.5 * l2_ev * seq_l2_ev)
        seq_grads = T.grad(seq_costs, self.params)
        seq_updates = [(par, par - lr * gra) for par, gra in zip(self.params, seq_grads)]
        update_x = T.set_subtensor(uiq_x, uiq_x - lr * T.grad(seq_costs, self.lt)[uiq_pqs])
        seq_updates.append((self.lt, update_x))     # 会直接更改到seq_updates里
        # ----------------------------------------------------------------------------

        # 输入正、负样本序列及其它参数后，更新变量，返回损失。
        uidx = T.iscalar()                              # T.iscalar()类型是 TensorType(int32, )
        self.seq_train = theano.function(
            inputs=[uidx],
            outputs=-upq + ae,
            updates=seq_updates,
            givens={
                pidxs: self.tra_buys_masks[uidx],       # 类型是 TensorType(int32, matrix)
                qidxs: self.tra_buys_neg_masks[uidx],
                tra_mask: self.tra_masks[uidx]})

    def train(self, idx):
        return self.seq_train(idx)

    def update_trained_items(self):
        # 获取图文融合表达。先用.eval()获得十进制数字，再用set_value()对shared变量做更新。
        mm = T.dot(self.fi, self.ei.T) + T.dot(self.ft, self.vt.T)    # shape=(n_item+1, 20)
        mm = mm.eval()
        self.mm.set_value(np.asarray(mm, dtype=theano.config.floatX), borrow=True)
        # 更新最终的items表达
        items = T.concatenate((self.lt, self.mm), axis=1)               # shape=(n_item+1, 40)
        items = items.eval()
        self.trained_items.set_value(np.asarray(items, dtype=theano.config.floatX), borrow=True)

    def update_trained_items2_corrupted_test_data(self):
        # update_trained_items: 用于train之后，重新用完整数据获得users表达。
        # update_trained_items2_corrupted_test: 用于test之前，获得test有部分缺失的items表达。
        # 获取图文融合表达。先用.eval()获得十进制数字，再用set_value()对shared变量做更新。
        mm = T.dot(self.fi_ct, self.ei.T) + T.dot(self.ft_ct, self.vt.T)    # shape=(n_item+1, 20)
        mm = mm.eval()
        self.mm.set_value(np.asarray(mm, dtype=theano.config.floatX), borrow=True)
        # 更新最终的items表达
        items = T.concatenate((self.lt, self.mm), axis=1)               # shape=(n_item+1, 40)
        items = items.eval()
        self.trained_items.set_value(np.asarray(items, dtype=theano.config.floatX), borrow=True)


# Obo的用不着，taobao太大。如果数据集小，可以用。
# ======================================================================================================================
class OboMvGru2Units(GruBasic2Units):
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_in, n_hidden,
                 n_img, n_txt, fea_img, fea_txt, fea_img_ct, fea_txt_ct):
        super(OboMvGru2Units, self).__init__(train, test, alpha_lambda, n_user, n_item, n_in, n_hidden)
        self.fi = theano.shared(borrow=True, value=np.asarray(fea_img, dtype=theano.config.floatX))  # shape=(n, 1024)
        self.ft = theano.shared(borrow=True, value=np.asarray(fea_txt, dtype=theano.config.floatX))  # shape=(n, 100)
        self.fi_ct = theano.shared(borrow=True, value=np.asarray(fea_img_ct, dtype=theano.config.floatX))  # shape=(n, 1024)
        self.ft_ct = theano.shared(borrow=True, value=np.asarray(fea_txt_ct, dtype=theano.config.floatX))  # shape=(n, 100)
        # 其它参数
        rang = 0.5
        ei = uniform(-rang, rang, (n_in, n_img))                # shape=(20, 1024)
        vt = uniform(-rang, rang, (n_in, n_txt))                # shape=(20, 100)
        # 建立参数。
        self.ei = theano.shared(borrow=True, value=ei.astype(theano.config.floatX))
        self.vt = theano.shared(borrow=True, value=vt.astype(theano.config.floatX))
        self.params = [
            self.uix, self.whx, self.bix,                        # self.lt单独进行更新。
            self.uim, self.whm, self.bim,
            self.ei, self.vt]
        self.l2_sqr = (
            T.sum(self.lt ** 2) +
            T.sum([T.sum(param ** 2) for param in self.params[:6]]))
        self.l2_ev = (
            T.sum([T.sum(param ** 2) for param in self.params[6:]]))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * self.l2_sqr +
            0.5 * self.alpha_lambda[2] * self.l2_ev)
        self.__theano_train__(n_hidden, n_img, n_txt)
        self.__theano_predict__(n_in, n_hidden)

    def __theano_train__(self, n_hidden, n_img, n_txt):
        """
        训练阶段跑一遍训练序列
        """
        # self.alpha_lambda = ['alpha', 'lambda', 'lambda_ev', 'lambda_ae', 'fea_random_zero']
        uix, whx = self.uix, self.whx
        uim, whm = self.uim, self.whm
        ei, vt = self.ei, self.vt

        tra_mask = T.ivector()
        seq_length = T.sum(tra_mask)                # 有效长度

        h0x, h0m = self.h0x, self.h0m
        bix, bim = self.bix, self.bim

        pidxs, qidxs = T.ivector(), T.ivector()
        xps, xqs = self.lt[pidxs], self.lt[qidxs]   # shape((seq_length, n_in))
        ips, iqs = self.fi[pidxs], self.fi[qidxs]   # shape((seq_length, n_img))
        tps, tqs = self.ft[pidxs], self.ft[qidxs]   # shape((seq_length, n_txt))

        pqs = T.concatenate((pidxs, qidxs))         # 先拼接
        uiq_pqs = Unique(False, False, False)(pqs)  # 再去重
        uiq_x = self.lt[uiq_pqs]                    # 相应的items特征

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
        zero = self.alpha_lambda[4]
        if 0.0 == zero:     # 用完全数据
            def recurrence(xp_t, xq_t, ip_t, iq_t, tp_t, tq_t, hx_t_pre1, hm_t_pre1):
                # item表达
                mp_t = T.dot(ei, ip_t) + T.dot(vt, tp_t)
                mq_t = T.dot(ei, iq_t) + T.dot(vt, tq_t)
                # 隐层计算
                z_rx = sigmoid(T.dot(uix[:2], xp_t) + T.dot(whx[:2], hx_t_pre1) + bix[:2])
                z_rm = sigmoid(T.dot(uim[:2], mp_t) + T.dot(whm[:2], hm_t_pre1) + bim[:2])
                zx, rx = z_rx[0], z_rx[1]
                zm, rm = z_rm[0], z_rm[1]
                cx = tanh(T.dot(uix[2], xp_t) + T.dot(whx[2], (rx * hx_t_pre1)) + bix[2])
                cm = tanh(T.dot(uim[2], mp_t) + T.dot(whm[2], (rm * hm_t_pre1)) + bim[2])
                hx_t = (T.ones_like(zx) - zx) * hx_t_pre1 + zx * cx
                hm_t = (T.ones_like(zm) - zm) * hm_t_pre1 + zm * cm
                # 偏好误差
                upq_t = (
                    T.dot(hx_t_pre1, xp_t - xq_t) +
                    T.dot(hm_t_pre1, mp_t - mq_t))
                loss_t = T.log(sigmoid(upq_t))
                # 重构误差
                loss_ae_t_i = (
                    T.sum((ip_t - T.dot(ei.T, mp_t)) ** 2) +
                    T.sum((iq_t - T.dot(ei.T, mq_t)) ** 2))
                loss_ae_t_t = (
                    T.sum((tp_t - T.dot(vt.T, mp_t)) ** 2) +
                    T.sum((tq_t - T.dot(vt.T, mq_t)) ** 2))
                return [hx_t, hm_t, loss_t, loss_ae_t_i, loss_ae_t_t]
            [hx, hm, loss, loss_ae_i, loss_ae_t], _ = theano.scan(
                fn=recurrence,
                sequences=[xps, xqs, ips, iqs, tps, tqs],
                outputs_info=[h0x, h0m, None, None, None],
                n_steps=seq_length,
                truncate_gradient=-1)
        else:
            # 每条序列训练前都随机whole feature corrupted
            ipsc = self.get_corrupted_input_whole(ips, zero)
            iqsc = self.get_corrupted_input_whole(iqs, zero)
            tpsc = self.get_corrupted_input_whole(tps, zero)
            tqsc = self.get_corrupted_input_whole(tqs, zero)

            def recurrence(xp_t, xq_t, ip_t, iq_t, tp_t, tq_t,
                           ipc_t, iqc_t, tpc_t, tqc_t, hx_t_pre1, hm_t_pre1):
                # item表达
                mp_t = T.dot(ei, ipc_t) + T.dot(vt, tpc_t)
                mq_t = T.dot(ei, iqc_t) + T.dot(vt, tqc_t)
                # 隐层计算
                z_rx = sigmoid(T.dot(uix[:2], xp_t) + T.dot(whx[:2], hx_t_pre1) + bix[:2])
                z_rm = sigmoid(T.dot(uim[:2], mp_t) + T.dot(whm[:2], hm_t_pre1) + bim[:2])
                zx, rx = z_rx[0], z_rx[1]
                zm, rm = z_rm[0], z_rm[1]
                cx = tanh(T.dot(uix[2], xp_t) + T.dot(whx[2], (rx * hx_t_pre1)) + bix[2])
                cm = tanh(T.dot(uim[2], mp_t) + T.dot(whm[2], (rm * hm_t_pre1)) + bim[2])
                hx_t = (T.ones_like(zx) - zx) * hx_t_pre1 + zx * cx
                hm_t = (T.ones_like(zm) - zm) * hm_t_pre1 + zm * cm
                # 偏好误差
                upq_t = (
                    T.dot(hx_t_pre1, xp_t - xq_t) +
                    T.dot(hm_t_pre1, mp_t - mq_t))
                loss_t = T.log(sigmoid(upq_t))
                # 重构误差
                loss_ae_t_i = (
                    T.sum((ip_t - T.dot(ei.T, mp_t)) ** 2) +
                    T.sum((iq_t - T.dot(ei.T, mq_t)) ** 2))
                loss_ae_t_t = (
                    T.sum((tp_t - T.dot(vt.T, mp_t)) ** 2) +
                    T.sum((tq_t - T.dot(vt.T, mq_t)) ** 2))
                return [hx_t, hm_t, loss_t, loss_ae_t_i, loss_ae_t_t]
            [hx, hm, loss, loss_ae_i, loss_ae_t], _ = theano.scan(
                fn=recurrence,
                sequences=[xps, xqs, ips, iqs, tps, tqs, ipsc, iqsc, tpsc, tqsc],
                outputs_info=[h0x, h0m, None, None, None],
                n_steps=seq_length,
                truncate_gradient=-1)

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, l2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        l2_ev = self.alpha_lambda[2]
        l2_ae = self.alpha_lambda[3]
        seq_l2_sq = T.sum([T.sum(par ** 2) for par in [xps, xqs, uix, uim, whx, whm, bix, bim]])
        seq_l2_ev = T.sum([T.sum(par ** 2) for par in [ei, vt]])
        upq = T.sum(loss)
        ae = (
            0.5 * l2_ae * T.sum(loss_ae_i) / n_img +
            0.5 * l2_ae * T.sum(loss_ae_t) / n_txt)
        seq_costs = (
            - upq + ae +
            0.5 * l2 * seq_l2_sq +
            0.5 * l2_ev * seq_l2_ev)
        seq_grads = T.grad(seq_costs, self.params)
        seq_updates = [(par, par - lr * gra) for par, gra in zip(self.params, seq_grads)]
        update_x = T.set_subtensor(uiq_x, uiq_x - lr * T.grad(seq_costs, self.lt)[uiq_pqs])
        seq_updates.append((self.lt, update_x))     # 会直接更改到seq_updates里
        # ----------------------------------------------------------------------------

        # 输入正、负样本序列及其它参数后，更新变量，返回损失。
        uidx = T.iscalar()                              # T.iscalar()类型是 TensorType(int32, )
        self.seq_train = theano.function(
            inputs=[uidx],
            outputs=-upq + ae,
            updates=seq_updates,
            givens={
                pidxs: self.tra_buys_masks[uidx],       # 类型是 TensorType(int32, matrix)
                qidxs: self.tra_buys_neg_masks[uidx],
                tra_mask: self.tra_masks[uidx]})

    def train(self, idx):
        return self.seq_train(idx)

    def update_trained_items(self):
        # 获取图文融合表达。先用.eval()获得十进制数字，再用set_value()对shared变量做更新。
        mm = T.dot(self.fi, self.ei.T) + T.dot(self.ft, self.vt.T)    # shape=(n_item+1, 20)
        mm = mm.eval()
        self.mm.set_value(np.asarray(mm, dtype=theano.config.floatX), borrow=True)
        # 更新最终的items表达
        items = T.concatenate((self.lt, self.mm), axis=1)               # shape=(n_item+1, 40)
        items = items.eval()
        self.trained_items.set_value(np.asarray(items, dtype=theano.config.floatX), borrow=True)

    def update_trained_items2_corrupted_test_data(self):
        # update_trained_items: 用于train之后，重新用完整数据获得users表达。
        # update_trained_items2_corrupted_test: 用于test之前，获得test有部分缺失的items表达。
        # 获取图文融合表达。先用.eval()获得十进制数字，再用set_value()对shared变量做更新。
        mm = T.dot(self.fi_ct, self.ei.T) + T.dot(self.ft_ct, self.vt.T)    # shape=(n_item+1, 20)
        mm = mm.eval()
        self.mm.set_value(np.asarray(mm, dtype=theano.config.floatX), borrow=True)
        # 更新最终的items表达
        items = T.concatenate((self.lt, self.mm), axis=1)               # shape=(n_item+1, 40)
        items = items.eval()
        self.trained_items.set_value(np.asarray(items, dtype=theano.config.floatX), borrow=True)


# denoising(把部分图文特征直接置为零)
# ======================================================================================================================
class MvGru1Unit(GruBasic):
    """
    1. 图文特征降维，用add做融合，再单独重构。 multi-modal
    2. 和latent特征做拼接.
    3. 送入1个GRU
    """
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_in, n_hidden,
                 n_img, n_txt, fea_img, fea_txt, fea_img_ct, fea_txt_ct):
        super(MvGru1Unit, self).__init__(train, test, alpha_lambda, n_user, n_item, n_in, n_hidden)
        self.fi = theano.shared(borrow=True, value=np.asarray(fea_img, dtype=theano.config.floatX))  # shape=(n, 1024)
        self.ft = theano.shared(borrow=True, value=np.asarray(fea_txt, dtype=theano.config.floatX))  # shape=(n, 100)
        self.fi_ct = theano.shared(borrow=True, value=np.asarray(fea_img_ct, dtype=theano.config.floatX))  # shape=(n, 1024)
        self.ft_ct = theano.shared(borrow=True, value=np.asarray(fea_txt_ct, dtype=theano.config.floatX))  # shape=(n, 100)
        # 其它参数
        rang = 0.5
        lt = uniform(-rang, rang, (n_item + 1, n_in))   # 多出来一个(填充符)，存放用于补齐用户购买序列/实际不存在的item
        mm = uniform(-rang, rang, (n_item + 1, n_in))   # 低维的多模态融合特征，和 lt 一一对应。
        ei = uniform(-rang, rang, (n_in, n_img))                # shape=(20, 1024)
        vt = uniform(-rang, rang, (n_in, n_txt))                # shape=(20, 100)
        self.lt = theano.shared(borrow=True, value=lt.astype(theano.config.floatX))
        self.mm = theano.shared(borrow=True, value=mm.astype(theano.config.floatX))
        self.ei = theano.shared(borrow=True, value=ei.astype(theano.config.floatX))
        self.vt = theano.shared(borrow=True, value=vt.astype(theano.config.floatX))
        self.params = [
            self.ui, self.wh, self.bi,                  # self.lt单独进行更新。
            self.ei, self.vt]
        self.l2_sqr = (
            T.sum(self.lt ** 2) +
            T.sum([T.sum(param ** 2) for param in self.params[:3]]))
        self.l2_ev = (
            T.sum([T.sum(param ** 2) for param in self.params[3:]]))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * self.l2_sqr +
            0.5 * self.alpha_lambda[2] * self.l2_ev)
        self.__theano_train__(n_in, n_hidden, n_img, n_txt)
        self.__theano_predict__(n_in, n_hidden)

    def __theano_train__(self, n_in, n_hidden, n_img, n_txt):
        """
        训练阶段跑一遍训练序列
        """
        # self.alpha_lambda = ['alpha', 'lambda', 'lambda_ev', 'lambda_ae', 'fea_random_zero']
        ui, wh = self.ui, self.wh
        ei, vt = self.ei, self.vt

        tra_mask = T.imatrix()                          # shape=(n, 157)
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))     # 获取mini-batch里各序列的长度最大值作为seq_length
        mask = tra_mask.T                               # shape=(157, n)

        h0 = T.alloc(self.h0, actual_batch_size, n_hidden)      # shape=(n, 40)
        bi = T.alloc(self.bi, actual_batch_size, 3, n_hidden)   # shape=(n, 3, 40), n_hidden放在最后
        bi = bi.dimshuffle(1, 2, 0)                             # shape=(3, 40, n)

        pidxs, qidxs = T.imatrix(), T.imatrix()         # TensorType(int32, matrix)
        xps, xqs = self.lt[pidxs], self.lt[qidxs]       # shape((actual_batch_size, seq_length, n_in))
        ips, iqs = self.fi[pidxs], self.fi[qidxs]       # shape((actual_batch_size, seq_length, n_img))
        tps, tqs = self.ft[pidxs], self.ft[qidxs]       # shape((actual_batch_size, seq_length, n_txt))
        xps, xqs = xps.dimshuffle(1, 0, 2), xqs.dimshuffle(1, 0, 2)     # shape=(seq_len, batch_size, n_in)
        ips, iqs = ips.dimshuffle(1, 0, 2), iqs.dimshuffle(1, 0, 2)
        tps, tqs = tps.dimshuffle(1, 0, 2), tqs.dimshuffle(1, 0, 2)

        pqs = T.concatenate((pidxs, qidxs))         # 先拼接
        uiq_pqs = Unique(False, False, False)(pqs)  # 再去重
        uiq_x = self.lt[uiq_pqs]                    # 相应的items特征

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
        zero = self.alpha_lambda[4]
        if 0.0 == zero:     # 用完全数据
            def recurrence(xp_t, xq_t, ip_t, iq_t, tp_t, tq_t, mask_t, h_t_pre1):
                # item表达
                mp_t = T.dot(ip_t, ei.T) + T.dot(tp_t, vt.T)    # shape=(n, 20)
                mq_t = T.dot(iq_t, ei.T) + T.dot(tq_t, vt.T)
                p_t = T.concatenate((xp_t, mp_t), axis=1)       # shape=(n, 40)
                q_t = T.concatenate((xq_t, mq_t), axis=1)
                # 隐层计算
                z_r = sigmoid(T.dot(ui[:2], p_t.T) +
                              T.dot(wh[:2], h_t_pre1.T) + bi[:2])
                z, r = z_r[0].T, z_r[1].T           # shape=(n, 40)
                c = tanh(T.dot(ui[2], p_t.T) +
                         T.dot(wh[2], (r * h_t_pre1).T) + bi[2])
                h_t = (T.ones_like(z) - z) * h_t_pre1 + z * c.T
                # 偏好误差
                upq_t = T.sum(h_t_pre1 * (p_t - q_t), axis=1)   # shape=(n, )
                loss_t = T.log(sigmoid(upq_t))                  # shape=(n, )
                loss_t *= mask_t
                # 重构误差
                loss_ae_t_i = (
                    T.sum((ip_t - T.dot(mp_t, ei)) ** 2) +
                    T.sum((iq_t - T.dot(mq_t, ei)) ** 2))     # T.sum(shape=(n, 1024), axis=1), 最后shape=(n,)
                loss_ae_t_t = (
                    T.sum((tp_t - T.dot(mp_t, vt)) ** 2) +
                    T.sum((tq_t - T.dot(mq_t, vt)) ** 2))
                loss_ae_t_i *= mask_t
                loss_ae_t_t *= mask_t
                return [h_t, loss_t, loss_ae_t_i, loss_ae_t_t]  # shape=(n, 20), (n, ), (n, )
            [h, loss, loss_ae_i, loss_ae_t], _ = theano.scan(
                fn=recurrence,
                sequences=[xps, xqs, ips, iqs, tps, tqs, mask],
                outputs_info=[h0, None, None, None],
                n_steps=seq_length,
                truncate_gradient=-1)
        else:
            # 每条序列训练前都随机whole feature corrupted
            ipsc = self.get_corrupted_input_whole_minibatch(ips, zero)
            iqsc = self.get_corrupted_input_whole_minibatch(iqs, zero)
            tpsc = self.get_corrupted_input_whole_minibatch(tps, zero)
            tqsc = self.get_corrupted_input_whole_minibatch(tqs, zero)

            def recurrence(xp_t, xq_t, ip_t, iq_t, tp_t, tq_t,
                           ipc_t, iqc_t, tpc_t, tqc_t, mask_t, h_t_pre1):
                # item表达
                mp_t = T.dot(ipc_t, ei.T) + T.dot(tpc_t, vt.T)    # shape=(n, 20)
                mq_t = T.dot(iqc_t, ei.T) + T.dot(tqc_t, vt.T)
                p_t = T.concatenate((xp_t, mp_t), axis=1)       # shape=(n, 40)
                q_t = T.concatenate((xq_t, mq_t), axis=1)
                # 隐层计算
                z_r = sigmoid(T.dot(ui[:2], p_t.T) +
                              T.dot(wh[:2], h_t_pre1.T) + bi[:2])
                z, r = z_r[0].T, z_r[1].T           # shape=(n, 40)
                c = tanh(T.dot(ui[2], p_t.T) +
                         T.dot(wh[2], (r * h_t_pre1).T) + bi[2])
                h_t = (T.ones_like(z) - z) * h_t_pre1 + z * c.T
                # 偏好误差
                upq_t = T.sum(h_t_pre1 * (p_t - q_t), axis=1)   # shape=(n, )
                loss_t = T.log(sigmoid(upq_t))                  # shape=(n, )
                loss_t *= mask_t
                # 重构误差
                loss_ae_t_i = (
                    T.sum((ip_t - T.dot(mp_t, ei)) ** 2) +
                    T.sum((iq_t - T.dot(mq_t, ei)) ** 2))     # T.sum(shape=(n, 1024), axis=1), 最后shape=(n,)
                loss_ae_t_t = (
                    T.sum((tp_t - T.dot(mp_t, vt)) ** 2) +
                    T.sum((tq_t - T.dot(mq_t, vt)) ** 2))
                loss_ae_t_i *= mask_t
                loss_ae_t_t *= mask_t
                return [h_t, loss_t, loss_ae_t_i, loss_ae_t_t]  # shape=(n, 20), (n, ), (n, )
            [h, loss, loss_ae_i, loss_ae_t], _ = theano.scan(
                fn=recurrence,
                sequences=[xps, xqs, ips, iqs, tps, tqs, ipsc, iqsc, tpsc, tqsc, mask],
                outputs_info=[h0, None, None, None],
                n_steps=seq_length,
                truncate_gradient=-1)

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, l2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        l2_ev = self.alpha_lambda[2]
        l2_ae = self.alpha_lambda[3]
        seq_l2_sq = (
            T.sum([T.sum(par ** 2) for par in [xps, xqs, ui, wh]]) +
            T.sum([T.sum(par ** 2) for par in [bi]]) / actual_batch_size)
        seq_l2_ev = (
            T.sum([T.sum(par ** 2) for par in [ei, vt]]))
        upq = T.sum(loss)
        ae = (
            0.5 * l2_ae * T.sum(loss_ae_i) / n_img +
            0.5 * l2_ae * T.sum(loss_ae_t) / n_txt)
        seq_costs = (
            (- upq + ae) / actual_batch_size +
            0.5 * l2 * seq_l2_sq +
            0.5 * l2_ev * seq_l2_ev)
        seq_grads = T.grad(seq_costs, self.params)
        seq_updates = [(par, par - lr * gra) for par, gra in zip(self.params, seq_grads)]
        update_x = T.set_subtensor(uiq_x, uiq_x - lr * T.grad(seq_costs, self.lt)[uiq_pqs])
        seq_updates.append((self.lt, update_x))     # 会直接更改到seq_updates里
        # ----------------------------------------------------------------------------

        # 输入正、负样本序列及其它参数后，更新变量，返回损失。
        # givens给数据
        start_end = T.ivector()
        self.seq_train = theano.function(
            inputs=[start_end],
            outputs=-upq + ae,
            updates=seq_updates,
            givens={
                pidxs: self.tra_buys_masks[start_end],       # 类型是 TensorType(int32, matrix)
                qidxs: self.tra_buys_neg_masks[start_end],   # T.ivector()类型是 TensorType(int32, vector)
                tra_mask: self.tra_masks[start_end]})

    def train(self, idxs):
        return self.seq_train(idxs)

    def update_trained_items(self):
        # 获取图文融合表达。先用.eval()获得十进制数字，再用set_value()对shared变量做更新。
        mm = T.dot(self.fi, self.ei.T) + T.dot(self.ft, self.vt.T)    # shape=(n_item+1, 20)
        mm = mm.eval()
        self.mm.set_value(np.asarray(mm, dtype=theano.config.floatX), borrow=True)
        # 更新最终的items表达
        items = T.concatenate((self.lt, self.mm), axis=1)               # shape=(n_item+1, 40)
        items = items.eval()
        self.trained_items.set_value(np.asarray(items, dtype=theano.config.floatX), borrow=True)

    def update_trained_items2_corrupted_test_data(self):
        # update_trained_items: 用于train之后，重新用完整数据获得users表达。
        # update_trained_items2_corrupted_test: 用于test之前，获得test有部分缺失的items表达。
        # 获取图文融合表达。先用.eval()获得十进制数字，再用set_value()对shared变量做更新。
        mm = T.dot(self.fi_ct, self.ei.T) + T.dot(self.ft_ct, self.vt.T)    # shape=(n_item+1, 20)
        mm = mm.eval()
        self.mm.set_value(np.asarray(mm, dtype=theano.config.floatX), borrow=True)
        # 更新最终的items表达
        items = T.concatenate((self.lt, self.mm), axis=1)               # shape=(n_item+1, 40)
        items = items.eval()
        self.trained_items.set_value(np.asarray(items, dtype=theano.config.floatX), borrow=True)


# denoising(把部分图文特征直接置为零)
# ======================================================================================================================
class MvGru2Units(GruBasic2Units):
    """
    1. 图文特征降维，用add做融合，再单独重构。 multi-modal
    2. 和latent特征 不 做拼接.
    3. 送入2个GRU
    """
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_in, n_hidden,
                 n_img, n_txt, fea_img, fea_txt, fea_img_ct, fea_txt_ct):
        super(MvGru2Units, self).__init__(train, test, alpha_lambda, n_user, n_item, n_in, n_hidden)
        self.fi = theano.shared(borrow=True, value=np.asarray(fea_img, dtype=theano.config.floatX))  # shape=(n, 1024)
        self.ft = theano.shared(borrow=True, value=np.asarray(fea_txt, dtype=theano.config.floatX))  # shape=(n, 100)
        self.fi_ct = theano.shared(borrow=True, value=np.asarray(fea_img_ct, dtype=theano.config.floatX))  # shape=(n, 1024)
        self.ft_ct = theano.shared(borrow=True, value=np.asarray(fea_txt_ct, dtype=theano.config.floatX))  # shape=(n, 100)
        # 其它参数
        rang = 0.5
        ei = uniform(-rang, rang, (n_in, n_img))                # shape=(20, 1024)
        vt = uniform(-rang, rang, (n_in, n_txt))                # shape=(20, 100)
        # 建立参数。
        self.ei = theano.shared(borrow=True, value=ei.astype(theano.config.floatX))
        self.vt = theano.shared(borrow=True, value=vt.astype(theano.config.floatX))
        self.params = [
            self.uix, self.whx, self.bix,    # self.lt单独进行更新。
            self.uim, self.whm, self.bim,
            self.ei, self.vt]
        self.l2_sqr = (
            T.sum(self.lt ** 2) +
            T.sum([T.sum(param ** 2) for param in self.params[:6]]))
        self.l2_ev = (
            T.sum([T.sum(param ** 2) for param in self.params[6:]]))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * self.l2_sqr +
            0.5 * self.alpha_lambda[2] * self.l2_ev)
        self.__theano_train__(n_in, n_hidden, n_img, n_txt)
        self.__theano_predict__(n_in, n_hidden)

    def __theano_train__(self, n_in, n_hidden, n_img, n_txt):
        """
        训练阶段跑一遍训练序列
        """
        # self.alpha_lambda = ['alpha', 'lambda', 'lambda_ev', 'lambda_ae', 'fea_random_zero']
        uix, whx = self.uix, self.whx
        uim, whm = self.uim, self.whm
        ei, vt = self.ei, self.vt

        tra_mask = T.imatrix()                          # shape=(n, 157)
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))     # 获取mini-batch里各序列的长度最大值作为seq_length
        mask = tra_mask.T                               # shape=(157, n)

        h0x = T.alloc(self.h0x, actual_batch_size, n_hidden)      # shape=(n, 20)
        h0m = T.alloc(self.h0m, actual_batch_size, n_hidden)
        bix = T.alloc(self.bix, actual_batch_size, 3, n_hidden)   # shape=(n, 3, 20), 原维度放在后边
        bim = T.alloc(self.bim, actual_batch_size, 3, n_hidden)   # shape=(n, 3, 20), 原维度放在后边
        bix = bix.dimshuffle(1, 2, 0)                             # shape=(3, 20, n)
        bim = bim.dimshuffle(1, 2, 0)                             # shape=(3, 20, n)

        pidxs, qidxs = T.imatrix(), T.imatrix()         # TensorType(int32, matrix)
        xps, xqs = self.lt[pidxs], self.lt[qidxs]       # shape((actual_batch_size, seq_length, n_in))
        ips, iqs = self.fi[pidxs], self.fi[qidxs]       # shape((actual_batch_size, seq_length, n_img))
        tps, tqs = self.ft[pidxs], self.ft[qidxs]       # shape((actual_batch_size, seq_length, n_txt))
        xps, xqs = xps.dimshuffle(1, 0, 2), xqs.dimshuffle(1, 0, 2)     # shape=(seq_len, batch_size, n_in)
        ips, iqs = ips.dimshuffle(1, 0, 2), iqs.dimshuffle(1, 0, 2)
        tps, tqs = tps.dimshuffle(1, 0, 2), tqs.dimshuffle(1, 0, 2)

        pqs = T.concatenate((pidxs, qidxs))         # 先拼接
        uiq_pqs = Unique(False, False, False)(pqs)  # 再去重
        uiq_x = self.lt[uiq_pqs]                    # 相应的items特征

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
        zero = self.alpha_lambda[4]
        if 0.0 == zero:     # 用完全数据
            def recurrence(xp_t, xq_t, ip_t, iq_t, tp_t, tq_t, mask_t, hx_t_pre1, hm_t_pre1):
                # item表达
                mp_t = T.dot(ip_t, ei.T) + T.dot(tp_t, vt.T)    # shape=(n, 20)
                mq_t = T.dot(iq_t, ei.T) + T.dot(tq_t, vt.T)
                # 隐层计算
                z_rx = sigmoid(T.dot(uix[:2], xp_t.T) + T.dot(whx[:2], hx_t_pre1.T) + bix[:2])
                z_rm = sigmoid(T.dot(uim[:2], mp_t.T) + T.dot(whm[:2], hm_t_pre1.T) + bim[:2])
                zx, rx = z_rx[0].T, z_rx[1].T                   # shape=(n, 20)
                zm, rm = z_rm[0].T, z_rm[1].T
                cx = tanh(T.dot(uix[2], xp_t.T) + T.dot(whx[2], (rx * hx_t_pre1).T) + bix[2])
                cm = tanh(T.dot(uim[2], mp_t.T) + T.dot(whm[2], (rm * hm_t_pre1).T) + bim[2])
                hx_t = (T.ones_like(zx) - zx) * hx_t_pre1 + zx * cx.T
                hm_t = (T.ones_like(zm) - zm) * hm_t_pre1 + zm * cm.T
                # 偏好误差
                upq_t = (
                    T.sum(hx_t_pre1 * (xp_t - xq_t), axis=1) +
                    T.sum(hm_t_pre1 * (mp_t - mq_t), axis=1))   # shape=(n, )
                loss_t = T.log(sigmoid(upq_t))
                loss_t *= mask_t
                # 重构误差
                loss_ae_t_i = (
                    T.sum((ip_t - T.dot(mp_t, ei)) ** 2) +
                    T.sum((iq_t - T.dot(mq_t, ei)) ** 2))     # T.sum(shape=(n, 1024), axis=1), 最后shape=(n,)
                loss_ae_t_t = (
                    T.sum((tp_t - T.dot(mp_t, vt)) ** 2) +
                    T.sum((tq_t - T.dot(mq_t, vt)) ** 2))
                loss_ae_t_i *= mask_t
                loss_ae_t_t *= mask_t
                return [hx_t, hm_t, loss_t, loss_ae_t_i, loss_ae_t_t]
            [hx, hm, loss, loss_ae_i, loss_ae_t], _ = theano.scan(
                fn=recurrence,
                sequences=[xps, xqs, ips, iqs, tps, tqs, mask],
                outputs_info=[h0x, h0m, None, None, None],
                n_steps=seq_length,
                truncate_gradient=-1)
        else:
            # 每条序列训练前都随机whole feature corrupted
            ipsc = self.get_corrupted_input_whole_minibatch(ips, zero)
            iqsc = self.get_corrupted_input_whole_minibatch(iqs, zero)
            tpsc = self.get_corrupted_input_whole_minibatch(tps, zero)
            tqsc = self.get_corrupted_input_whole_minibatch(tqs, zero)

            def recurrence(xp_t, xq_t, ip_t, iq_t, tp_t, tq_t,
                           ipc_t, iqc_t, tpc_t, tqc_t, mask_t, hx_t_pre1, hm_t_pre1):
                # item表达
                mp_t = T.dot(ipc_t, ei.T) + T.dot(tpc_t, vt.T)    # shape=(n, 20)
                mq_t = T.dot(iqc_t, ei.T) + T.dot(tqc_t, vt.T)
                # 隐层计算
                z_rx = sigmoid(T.dot(uix[:2], xp_t.T) + T.dot(whx[:2], hx_t_pre1.T) + bix[:2])
                z_rm = sigmoid(T.dot(uim[:2], mp_t.T) + T.dot(whm[:2], hm_t_pre1.T) + bim[:2])
                zx, rx = z_rx[0].T, z_rx[1].T                   # shape=(n, 20)
                zm, rm = z_rm[0].T, z_rm[1].T
                cx = tanh(T.dot(uix[2], xp_t.T) + T.dot(whx[2], (rx * hx_t_pre1).T) + bix[2])
                cm = tanh(T.dot(uim[2], mp_t.T) + T.dot(whm[2], (rm * hm_t_pre1).T) + bim[2])
                hx_t = (T.ones_like(zx) - zx) * hx_t_pre1 + zx * cx.T
                hm_t = (T.ones_like(zm) - zm) * hm_t_pre1 + zm * cm.T
                # 偏好误差
                upq_t = (
                    T.sum(hx_t_pre1 * (xp_t - xq_t), axis=1) +
                    T.sum(hm_t_pre1 * (mp_t - mq_t), axis=1))   # shape=(n, )
                loss_t = T.log(sigmoid(upq_t))
                loss_t *= mask_t
                # 重构误差
                # denoising: 重构时用ip_t。
                # missing:   重构时用ipc_t。
                loss_ae_t_i = (
                    T.sum((ip_t - T.dot(mp_t, ei)) ** 2) +
                    T.sum((iq_t - T.dot(mq_t, ei)) ** 2))     # T.sum(shape=(n, 1024), axis=1), 最后shape=(n,)
                loss_ae_t_t = (
                    T.sum((tp_t - T.dot(mp_t, vt)) ** 2) +
                    T.sum((tq_t - T.dot(mq_t, vt)) ** 2))
                loss_ae_t_i *= mask_t
                loss_ae_t_t *= mask_t
                return [hx_t, hm_t, loss_t, loss_ae_t_i, loss_ae_t_t]
            [hx, hm, loss, loss_ae_i, loss_ae_t], _ = theano.scan(
                fn=recurrence,
                sequences=[xps, xqs, ips, iqs, tps, tqs, ipsc, iqsc, tpsc, tqsc, mask],
                outputs_info=[h0x, h0m, None, None, None],
                n_steps=seq_length,
                truncate_gradient=-1)

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, l2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        l2_ev = self.alpha_lambda[2]
        l2_ae = self.alpha_lambda[3]
        seq_l2_sq = (
            T.sum([T.sum(par ** 2) for par in [xps, xqs, uix, whx, uim, whm]]) +
            T.sum([T.sum(par ** 2) for par in [bix, bim]]) / actual_batch_size)
        seq_l2_ev = (
            T.sum([T.sum(par ** 2) for par in [ei, vt]]))
        upq = T.sum(loss)
        ae = (
            0.5 * l2_ae * T.sum(loss_ae_i) / n_img +
            0.5 * l2_ae * T.sum(loss_ae_t) / n_txt)
        seq_costs = (
            (- upq + ae) / actual_batch_size +
            0.5 * l2 * seq_l2_sq +
            0.5 * l2_ev * seq_l2_ev)
        seq_grads = T.grad(seq_costs, self.params)
        seq_updates = [(par, par - lr * gra) for par, gra in zip(self.params, seq_grads)]
        update_x = T.set_subtensor(uiq_x, uiq_x - lr * T.grad(seq_costs, self.lt)[uiq_pqs])
        seq_updates.append((self.lt, update_x))     # 会直接更改到seq_updates里
        # ----------------------------------------------------------------------------

        # 输入正、负样本序列及其它参数后，更新变量，返回损失。
        start_end = T.ivector()
        self.seq_train = theano.function(
            inputs=[start_end],
            outputs=-upq + ae,
            updates=seq_updates,
            givens={
                pidxs: self.tra_buys_masks[start_end],       # 类型是 TensorType(int32, matrix)
                qidxs: self.tra_buys_neg_masks[start_end],   # T.ivector()类型是 TensorType(int32, vector)
                tra_mask: self.tra_masks[start_end]})

    def train(self, idxs):
        return self.seq_train(idxs)

    def update_trained_items(self):
        # 获取图文融合表达。先用.eval()获得十进制数字，再用set_value()对shared变量做更新。
        mm = T.dot(self.fi, self.ei.T) + T.dot(self.ft, self.vt.T)    # shape=(n_item+1, 20)
        mm = mm.eval()
        self.mm.set_value(np.asarray(mm, dtype=theano.config.floatX), borrow=True)
        # 更新最终的items表达
        items = T.concatenate((self.lt, self.mm), axis=1)               # shape=(n_item+1, 40)
        items = items.eval()
        self.trained_items.set_value(np.asarray(items, dtype=theano.config.floatX), borrow=True)

    def update_trained_items2_corrupted_test_data(self):
        # update_trained_items: 用于train之后，重新用完整数据获得users表达。
        # update_trained_items2_corrupted_test: 用于test之前，获得test有部分缺失的items表达。
        # 获取图文融合表达。先用.eval()获得十进制数字，再用set_value()对shared变量做更新。
        mm = T.dot(self.fi_ct, self.ei.T) + T.dot(self.ft_ct, self.vt.T)    # shape=(n_item+1, 20)
        mm = mm.eval()
        self.mm.set_value(np.asarray(mm, dtype=theano.config.floatX), borrow=True)
        # 更新最终的items表达
        items = T.concatenate((self.lt, self.mm), axis=1)               # shape=(n_item+1, 40)
        items = items.eval()
        self.trained_items.set_value(np.asarray(items, dtype=theano.config.floatX), borrow=True)


# 不做denoising实验
# ======================================================================================================================
class MvGruCon(GruBasic):
    """
    1. 图文特征降维
    2. 和latent特征做拼接. Concatenate
    3. 送入1个GRU
    """
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_in, n_hidden,
                 n_img, n_txt, fea_img, fea_txt, fea_img_ct, fea_txt_ct):
        super(MvGruCon, self).__init__(train, test, alpha_lambda, n_user, n_item, n_in, n_hidden)
        self.fi = theano.shared(borrow=True, value=np.asarray(fea_img, dtype=theano.config.floatX))  # shape=(n, 1024)
        self.ft = theano.shared(borrow=True, value=np.asarray(fea_txt, dtype=theano.config.floatX))  # shape=(n, 100)
        self.fi_ct = theano.shared(borrow=True, value=np.asarray(fea_img_ct, dtype=theano.config.floatX))  # shape=(n, 1024)
        self.ft_ct = theano.shared(borrow=True, value=np.asarray(fea_txt_ct, dtype=theano.config.floatX))  # shape=(n, 100)
        # 其它参数
        rang = 0.5
        mi = uniform(-rang, rang, (n_item + 1, n_in))   # 图像的低维特征，和 lt 一一对应。
        mt = uniform(-rang, rang, (n_item + 1, n_in))   # 文本的低维特征，
        ei = uniform(-rang, rang, (n_in, n_img))                # shape=(20, 1024)
        vt = uniform(-rang, rang, (n_in, n_txt))                # shape=(20, 100)
        self.mi = theano.shared(borrow=True, value=mi.astype(theano.config.floatX))
        self.mt = theano.shared(borrow=True, value=mt.astype(theano.config.floatX))
        self.ei = theano.shared(borrow=True, value=ei.astype(theano.config.floatX))
        self.vt = theano.shared(borrow=True, value=vt.astype(theano.config.floatX))
        self.params = [
            self.ui, self.wh, self.bi,                  # self.lt单独进行更新。
            self.ei, self.vt]    # 只是拼接而已，不需要加bias
        self.l2_sqr = (
            T.sum(self.lt ** 2) +
            T.sum([T.sum(param ** 2) for param in self.params[:4]]))
        self.l2_ev = (
            T.sum(self.ei ** 2) +
            T.sum(self.vt ** 2))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * self.l2_sqr +
            0.5 * self.alpha_lambda[2] * self.l2_ev)
        self.__theano_train__(n_in, n_hidden, n_img, n_txt)
        self.__theano_predict__(n_in, n_hidden)

    def __theano_train__(self, n_in, n_hidden, n_img, n_txt):
        """
        训练阶段跑一遍训练序列
        """
        # self.alpha_lambda = ['alpha', 'lambda', 'lambda_ev']
        ui, wh = self.ui, self.wh
        ei, vt = self.ei, self.vt

        tra_mask = T.imatrix()                          # shape=(n, 157)
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))     # 获取mini-batch里各序列的长度最大值作为seq_length
        mask = tra_mask.T                               # shape=(157, n)

        h0 = T.alloc(self.h0, actual_batch_size, n_hidden)      # shape=(n, 20)
        bi = T.alloc(self.bi, actual_batch_size, 3, n_hidden)   # shape=(n, 3, 20), n_hidden放在最后
        bi = bi.dimshuffle(1, 2, 0)                             # shape=(3, 20, n)

        pidxs, qidxs = T.imatrix(), T.imatrix()         # TensorType(int32, matrix)
        xps, xqs = self.lt[pidxs], self.lt[qidxs]       # shape((actual_batch_size, seq_length, n_in))
        ips, iqs = self.fi[pidxs], self.fi[qidxs]       # shape((actual_batch_size, seq_length, n_img))
        tps, tqs = self.ft[pidxs], self.ft[qidxs]       # shape((actual_batch_size, seq_length, n_txt))
        xps, xqs = xps.dimshuffle(1, 0, 2), xqs.dimshuffle(1, 0, 2)     # shape=(seq_len, batch_size, n_in)
        ips, iqs = ips.dimshuffle(1, 0, 2), iqs.dimshuffle(1, 0, 2)
        tps, tqs = tps.dimshuffle(1, 0, 2), tqs.dimshuffle(1, 0, 2)

        pqs = T.concatenate((pidxs, qidxs))         # 先拼接
        uiq_pqs = Unique(False, False, False)(pqs)  # 再去重
        uiq_x = self.lt[uiq_pqs]                    # 相应的items特征

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
        def recurrence(xp_t, xq_t, ip_t, iq_t, tp_t, tq_t, mask_t, h_t_pre1):
            # item表达
            mip_t, mtp_t = T.dot(ip_t, ei.T), T.dot(tp_t, vt.T)     # shape=(n, 20)
            miq_t, mtq_t = T.dot(iq_t, ei.T), T.dot(tq_t, vt.T)
            p_t = T.concatenate((xp_t, mip_t, mtp_t), axis=1)       # shape=(n, 60)
            q_t = T.concatenate((xq_t, miq_t, mtq_t), axis=1)
            # 隐层计算
            z_r = sigmoid(T.dot(ui[:2], p_t.T) +
                          T.dot(wh[:2], h_t_pre1.T) + bi[:2])
            z, r = z_r[0].T, z_r[1].T           # shape=(n, 40)
            c = tanh(T.dot(ui[2], p_t.T) +
                     T.dot(wh[2], (r * h_t_pre1).T) + bi[2])
            h_t = (T.ones_like(z) - z) * h_t_pre1 + z * c.T
            # 偏好误差
            upq_t = T.sum(h_t_pre1 * (p_t - q_t), axis=1)   # shape=(n, )
            loss_t = T.log(sigmoid(upq_t))                  # shape=(n, )
            loss_t *= mask_t
            return [h_t, loss_t]  # shape=(n, 20), (n, ), (n, )
        [h, loss], _ = theano.scan(
            fn=recurrence,
            sequences=[xps, xqs, ips, iqs, tps, tqs, mask],
            outputs_info=[h0, None],
            n_steps=seq_length,
            truncate_gradient=-1)

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, l2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        l2_ev = self.alpha_lambda[2]
        seq_l2_sq = (
            T.sum([T.sum(par ** 2) for par in [xps, xqs, ui, wh]]) +
            T.sum([T.sum(par ** 2) for par in [bi]]) / actual_batch_size)
        seq_l2_ev = (
            T.sum([T.sum(par ** 2) for par in [ei, vt]]))
        upq = T.sum(loss)
        seq_costs = (
            - upq / actual_batch_size +
            0.5 * l2 * seq_l2_sq +
            0.5 * l2_ev * seq_l2_ev)
        seq_grads = T.grad(seq_costs, self.params)
        seq_updates = [(par, par - lr * gra) for par, gra in zip(self.params, seq_grads)]
        update_x = T.set_subtensor(uiq_x, uiq_x - lr * T.grad(seq_costs, self.lt)[uiq_pqs])
        seq_updates.append((self.lt, update_x))     # 会直接更改到seq_updates里
        # ----------------------------------------------------------------------------

        # 输入正、负样本序列及其它参数后，更新变量，返回损失。
        # givens给数据
        start_end = T.ivector()
        self.seq_train = theano.function(
            inputs=[start_end],
            outputs=-upq,
            updates=seq_updates,
            givens={
                pidxs: self.tra_buys_masks[start_end],       # 类型是 TensorType(int32, matrix)
                qidxs: self.tra_buys_neg_masks[start_end],   # T.ivector()类型是 TensorType(int32, vector)
                tra_mask: self.tra_masks[start_end]})

    def train(self, idxs):
        return self.seq_train(idxs)

    def update_trained_items(self):
        # 获取图文融合表达。先用.eval()获得十进制数字，再用set_value()对shared变量做更新。
        mi, mt = T.dot(self.fi, self.ei.T), T.dot(self.ft, self.vt.T)   # shape=(n_item+1, 20)
        mi, mt = mi.eval(), mt.eval()
        self.mi.set_value(np.asarray(mi, dtype=theano.config.floatX), borrow=True)
        self.mt.set_value(np.asarray(mt, dtype=theano.config.floatX), borrow=True)
        # 更新最终的items表达
        items = T.concatenate((self.lt, self.mi, self.mt), axis=1)      # shape=(n_item+1, 60)
        items = items.eval()
        self.trained_items.set_value(np.asarray(items, dtype=theano.config.floatX), borrow=True)

    def update_trained_items2_corrupted_test_data(self):
        # update_trained_items: 用于train之后，重新用完整数据获得users表达。
        # update_trained_items2_corrupted_test: 用于test之前，获得test有部分缺失的items表达。
        # 获取图文融合表达。先用.eval()获得十进制数字，再用set_value()对shared变量做更新。
        mi, mt = T.dot(self.fi_ct, self.ei.T), T.dot(self.ft_ct, self.vt.T)   # shape=(n_item+1, 20)
        mi, mt = mi.eval(), mt.eval()
        self.mi.set_value(np.asarray(mi, dtype=theano.config.floatX), borrow=True)
        self.mt.set_value(np.asarray(mt, dtype=theano.config.floatX), borrow=True)
        # 更新最终的items表达
        items = T.concatenate((self.lt, self.mi, self.mt), axis=1)      # shape=(n_item+1, 60)
        items = items.eval()
        self.trained_items.set_value(np.asarray(items, dtype=theano.config.floatX), borrow=True)


# 不做denoising实验
# ======================================================================================================================
class MvGruFusion(GruBasic):
    """
    1. 图文特征降维，用add做融合，再单独重构。 multi-modal
    2. 和latent特征做拼接.
    3. 送入1个GRU
    """
    def __init__(self, train, test, alpha_lambda, n_user, n_item, n_in, n_hidden,
                 n_img, n_txt, fea_img, fea_txt, fea_img_ct, fea_txt_ct):
        super(MvGruFusion, self).__init__(train, test, alpha_lambda, n_user, n_item, n_in, n_hidden)
        self.fi = theano.shared(borrow=True, value=np.asarray(fea_img, dtype=theano.config.floatX))  # shape=(n, 1024)
        self.ft = theano.shared(borrow=True, value=np.asarray(fea_txt, dtype=theano.config.floatX))  # shape=(n, 100)
        self.fi_ct = theano.shared(borrow=True, value=np.asarray(fea_img_ct, dtype=theano.config.floatX))  # shape=(n, 1024)
        self.ft_ct = theano.shared(borrow=True, value=np.asarray(fea_txt_ct, dtype=theano.config.floatX))  # shape=(n, 100)
        # 其它参数
        rang = 0.5
        lt = uniform(-rang, rang, (n_item + 1, n_in))   # 多出来一个(填充符)，存放用于补齐用户购买序列/实际不存在的item
        mm = uniform(-rang, rang, (n_item + 1, n_in))   # 低维的多模态融合特征，和 lt 一一对应。
        ei = uniform(-rang, rang, (n_in, n_img))                # shape=(20, 1024)
        vt = uniform(-rang, rang, (n_in, n_txt))                # shape=(20, 100)
        self.lt = theano.shared(borrow=True, value=lt.astype(theano.config.floatX))
        self.mm = theano.shared(borrow=True, value=mm.astype(theano.config.floatX))
        self.ei = theano.shared(borrow=True, value=ei.astype(theano.config.floatX))
        self.vt = theano.shared(borrow=True, value=vt.astype(theano.config.floatX))
        self.params = [
            self.ui, self.wh, self.bi,                  # self.lt单独进行更新。
            self.ei, self.vt]
        self.l2_sqr = (
            T.sum(self.lt ** 2) +
            T.sum([T.sum(param ** 2) for param in self.params[:3]]))
        self.l2_ev = (
            T.sum([T.sum(param ** 2) for param in self.params[3:]]))
        self.l2 = (
            0.5 * self.alpha_lambda[1] * self.l2_sqr +
            0.5 * self.alpha_lambda[2] * self.l2_ev)
        self.__theano_train__(n_in, n_hidden, n_img, n_txt)
        self.__theano_predict__(n_in, n_hidden)

    def __theano_train__(self, n_in, n_hidden, n_img, n_txt):
        """
        训练阶段跑一遍训练序列
        """
        # self.alpha_lambda = ['alpha', 'lambda', 'lambda_ev']
        ui, wh = self.ui, self.wh
        ei, vt = self.ei, self.vt

        tra_mask = T.imatrix()                          # shape=(n, 157)
        actual_batch_size = tra_mask.shape[0]
        seq_length = T.max(T.sum(tra_mask, axis=1))     # 获取mini-batch里各序列的长度最大值作为seq_length
        mask = tra_mask.T                               # shape=(157, n)

        h0 = T.alloc(self.h0, actual_batch_size, n_hidden)      # shape=(n, 40)
        bi = T.alloc(self.bi, actual_batch_size, 3, n_hidden)   # shape=(n, 3, 40), n_hidden放在最后
        bi = bi.dimshuffle(1, 2, 0)                             # shape=(3, 40, n)

        pidxs, qidxs = T.imatrix(), T.imatrix()         # TensorType(int32, matrix)
        xps, xqs = self.lt[pidxs], self.lt[qidxs]       # shape((actual_batch_size, seq_length, n_in))
        ips, iqs = self.fi[pidxs], self.fi[qidxs]       # shape((actual_batch_size, seq_length, n_img))
        tps, tqs = self.ft[pidxs], self.ft[qidxs]       # shape((actual_batch_size, seq_length, n_txt))
        xps, xqs = xps.dimshuffle(1, 0, 2), xqs.dimshuffle(1, 0, 2)     # shape=(seq_len, batch_size, n_in)
        ips, iqs = ips.dimshuffle(1, 0, 2), iqs.dimshuffle(1, 0, 2)
        tps, tqs = tps.dimshuffle(1, 0, 2), tqs.dimshuffle(1, 0, 2)

        pqs = T.concatenate((pidxs, qidxs))         # 先拼接
        uiq_pqs = Unique(False, False, False)(pqs)  # 再去重
        uiq_x = self.lt[uiq_pqs]                    # 相应的items特征

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
        def recurrence(xp_t, xq_t, ip_t, iq_t, tp_t, tq_t, mask_t, h_t_pre1):
            # item表达
            mp_t = T.dot(ip_t, ei.T) + T.dot(tp_t, vt.T)    # shape=(n, 20)
            mq_t = T.dot(iq_t, ei.T) + T.dot(tq_t, vt.T)
            p_t = T.concatenate((xp_t, mp_t), axis=1)       # shape=(n, 40)
            q_t = T.concatenate((xq_t, mq_t), axis=1)
            # 隐层计算
            z_r = sigmoid(T.dot(ui[:2], p_t.T) +
                          T.dot(wh[:2], h_t_pre1.T) + bi[:2])
            z, r = z_r[0].T, z_r[1].T           # shape=(n, 40)
            c = tanh(T.dot(ui[2], p_t.T) +
                     T.dot(wh[2], (r * h_t_pre1).T) + bi[2])
            h_t = (T.ones_like(z) - z) * h_t_pre1 + z * c.T
            # 偏好误差
            upq_t = T.sum(h_t_pre1 * (p_t - q_t), axis=1)   # shape=(n, )
            loss_t = T.log(sigmoid(upq_t))                  # shape=(n, )
            loss_t *= mask_t
            return [h_t, loss_t]  # shape=(n, 20), (n, ), (n, )
        [h, loss], _ = theano.scan(
            fn=recurrence,
            sequences=[xps, xqs, ips, iqs, tps, tqs, mask],
            outputs_info=[h0, None],
            n_steps=seq_length,
            truncate_gradient=-1)

        # ----------------------------------------------------------------------------
        # cost, gradients, learning rate, l2 regularization
        lr, l2 = self.alpha_lambda[0], self.alpha_lambda[1]
        l2_ev = self.alpha_lambda[2]
        seq_l2_sq = (
            T.sum([T.sum(par ** 2) for par in [xps, xqs, ui, wh]]) +
            T.sum([T.sum(par ** 2) for par in [bi]]) / actual_batch_size)
        seq_l2_ev = (
            T.sum([T.sum(par ** 2) for par in [ei, vt]]))
        upq = T.sum(loss)
        seq_costs = (
            - upq / actual_batch_size +
            0.5 * l2 * seq_l2_sq +
            0.5 * l2_ev * seq_l2_ev)
        seq_grads = T.grad(seq_costs, self.params)
        seq_updates = [(par, par - lr * gra) for par, gra in zip(self.params, seq_grads)]
        update_x = T.set_subtensor(uiq_x, uiq_x - lr * T.grad(seq_costs, self.lt)[uiq_pqs])
        seq_updates.append((self.lt, update_x))     # 会直接更改到seq_updates里
        # ----------------------------------------------------------------------------

        # 输入正、负样本序列及其它参数后，更新变量，返回损失。
        # givens给数据
        start_end = T.ivector()
        self.seq_train = theano.function(
            inputs=[start_end],
            outputs=-upq,
            updates=seq_updates,
            givens={
                pidxs: self.tra_buys_masks[start_end],       # 类型是 TensorType(int32, matrix)
                qidxs: self.tra_buys_neg_masks[start_end],   # T.ivector()类型是 TensorType(int32, vector)
                tra_mask: self.tra_masks[start_end]})

    def train(self, idxs):
        return self.seq_train(idxs)

    def update_trained_items(self):
        # 获取图文融合表达。先用.eval()获得十进制数字，再用set_value()对shared变量做更新。
        mm = T.dot(self.fi, self.ei.T) + T.dot(self.ft, self.vt.T)    # shape=(n_item+1, 20)
        mm = mm.eval()
        self.mm.set_value(np.asarray(mm, dtype=theano.config.floatX), borrow=True)
        # 更新最终的items表达
        items = T.concatenate((self.lt, self.mm), axis=1)               # shape=(n_item+1, 40)
        items = items.eval()
        self.trained_items.set_value(np.asarray(items, dtype=theano.config.floatX), borrow=True)

    def update_trained_items2_corrupted_test_data(self):
        # update_trained_items: 用于train之后，重新用完整数据获得users表达。
        # update_trained_items2_corrupted_test: 用于test之前，获得test有部分缺失的items表达。
        # 获取图文融合表达。先用.eval()获得十进制数字，再用set_value()对shared变量做更新。
        mm = T.dot(self.fi_ct, self.ei.T) + T.dot(self.ft_ct, self.vt.T)    # shape=(n_item+1, 20)
        mm = mm.eval()
        self.mm.set_value(np.asarray(mm, dtype=theano.config.floatX), borrow=True)
        # 更新最终的items表达
        items = T.concatenate((self.lt, self.mm), axis=1)               # shape=(n_item+1, 40)
        items = items.eval()
        self.trained_items.set_value(np.asarray(items, dtype=theano.config.floatX), borrow=True)


@exe_time
def main():
    print('... construct the class: MV-GRU')


if '__main__' == __name__:
    main()
