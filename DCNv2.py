#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 23:04:53 2020
@author: hu
"""
import tensorflow as tf
import keras.layers as KL

class DCNv2(KL.Layer):
    def __init__(self, filters, 
                 kernel_size, 
                 #stride, 
                 #padding, 
                 #dilation = 1, 
                 #deformable_groups = 1,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):
        #deformable_groups unsupported
        #dilation unsupported
        #stride unsupported
        #assert stride == 1
        #assert dilation == 1
        #assert deformable_groups == 1
        self.filters = filters
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (1, 1, 1, 1)
        #self.padding = padding
        self.dilation = (1, 1)
        self.deformable_groups = 1
        self.use_bias = use_bias
        self.kernel_initializer=kernel_initializer
        self.bias_initializer=bias_initializer
        self.kernel_regularizer=kernel_regularizer
        self.bias_regularizer=bias_regularizer
        super(DCNv2, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.kernel = self.add_weight(
            name = 'kernel',
            shape = self.kernel_size + (input_shape[-1].value, self.filters),
            initializer = self.kernel_initializer,
            regularizer = self.kernel_regularizer,
            trainable = True,
            dtype = 'float32',
            )
        
        if self.use_bias:
            self.bias = self.add_weight(
                name = 'bias',
                shape = (self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
                dtype='float32',
                )
        
        #[kh, kw, ic, 3 * groups * kh, kw]--->3 * groups * kh * kw = oc [output channels]
        self.offset_kernel = self.add_weight(
            name = 'offset_kernel',
            shape = self.kernel_size + (input_shape[-1], 3 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1]), 
            initializer = 'zeros',
            trainable = True,
            dtype = 'float32')
        
        self.offset_bias = self.add_weight(
            name = 'offset_bias',
            shape = (3 * self.kernel_size[0] * self.kernel_size[1] * self.deformable_groups,),
            initializer='zeros',
            trainable = True,
            dtype = 'float32',
            )
        self.ks = self.kernel_size[0] * self.kernel_size[1]
        self.ph, self.pw = (self.kernel_size[0] - 1) / 2, (self.kernel_size[1] - 1) / 2
        self.patch_x, self.patch_y = tf.meshgrid(tf.range(-self.pw, self.pw + 1), tf.range(-self.ph, self.ph + 1))
        self.patch_x, self.patch_y = tf.reshape(self.patch_x, [1, 1, 1, -1]), tf.reshape(self.patch_y, [1, 1, 1, -1])
        super(DCNv2, self).build(input_shape)
        
    def call(self, x):
        #x: [B, H, W, C]
        #offset: [B, H, W, ic] convx [kh, kw, ic, 3 * groups * kh * kw] ---> [B, H, W, 3 * groups * kh * kw]
        offset = tf.nn.conv2d(x, self.offset_kernel, strides = self.stride, padding = 'SAME')
        offset += self.offset_bias
        
        bs, ih, iw, ic = [v.value for v in x.shape]
        bs = tf.shape(x)[0]
        
        #[B, H, W, oc'] oc' = groups * kh * kw
        #other implementations organize data as: [yxyxyxyxyxyxyxyxyxmmmmmmmmm]
        oyox, mask = offset[..., :2*self.ks], offset[..., 2*self.ks:]
        oy, ox = tf.split(tf.reshape(oyox, [-1, 2]), 2, axis = 1)
        oy, ox = tf.reshape(oy, [bs, ih, iw, self.ks]), tf.reshape(ox, [bs, ih, iw, self.ks])
        
        #we prefer: [yyyyyyyyyxxxxxxxxxmmmmmmmmm]
        #oy, ox, mask = tf.split(offset, 3, axis = -1)
        mask = tf.nn.sigmoid(mask)
        
        #[H, W], [H, W]
        grid_x, grid_y = tf.meshgrid(tf.range(iw), tf.range(ih))
        grid_x, grid_y = tf.cast(grid_x, 'float32'), tf.cast(grid_y, 'float32')
        
        #[B, H, W, oc'], [B, H, W, oc']
        grid_x, grid_y = tf.tile(tf.expand_dims(tf.expand_dims(grid_x, axis = -1), axis = 0), [bs, 1, 1, 1]) + self.pw + self.patch_x, \
                         tf.tile(tf.expand_dims(tf.expand_dims(grid_y, axis = -1), axis = 0), [bs, 1, 1, 1]) + self.ph + self.patch_y
        
        #[B, H, W, oc']
        grid_x, grid_y = grid_x + ox, grid_y + oy
        
        grid_ix0, grid_iy0 = tf.floor(grid_x), tf.floor(grid_y)
        
        grid_ix1, grid_iy1 = tf.clip_by_value(grid_ix0 + 1, 0, tf.cast(iw, 'float32') + 1), tf.clip_by_value(grid_iy0 + 1, 0, tf.cast(ih, 'float32') + 1)

        grid_ix0, grid_iy0 = tf.clip_by_value(grid_ix0, 0, tf.cast(iw, 'float32') + 1), tf.clip_by_value(grid_iy0, 0, tf.cast(ih, 'float32') + 1)

        grid_x, grid_y = tf.clip_by_value(grid_x, 0, tf.cast(iw, 'float32') + 1), tf.clip_by_value(grid_y, 0, tf.cast(ih, 'float32') + 1)
        
        
        #weights
        w_00 = (grid_x - grid_ix0) * (grid_y - grid_iy0)
        w_01 = (grid_x - grid_ix0) * (grid_iy1 - grid_y)
        w_11 = (grid_ix1 - grid_x) * (grid_iy1 - grid_y)
        w_10 = (grid_ix1 - grid_x) * (grid_y - grid_iy0)
        
        #[B, 1]
        batch_index = tf.reshape(tf.range(bs), [bs, 1, 1, 1])
        #[B, H, W, oc']
        grid_batch = tf.ones_like(grid_ix0, dtype = 'int32') * batch_index
        
        #[B, H, W, oc', 3]
        grid_ix0, grid_iy0, grid_ix1, grid_iy1 = tf.cast(grid_ix0, 'int32'), \
                                                 tf.cast(grid_iy0, 'int32'), \
                                                 tf.cast(grid_ix1, 'int32'), \
                                                 tf.cast(grid_iy1, 'int32')
        grid_ix0iy0 = tf.stack([grid_batch, grid_iy0, grid_ix0], axis = -1)
        grid_ix0iy1 = tf.stack([grid_batch, grid_iy1, grid_ix0], axis = -1)
        grid_ix1iy1 = tf.stack([grid_batch, grid_iy1, grid_ix1], axis = -1)
        grid_ix1iy0 = tf.stack([grid_batch, grid_iy0, grid_ix1], axis = -1)
        
        #[B, H + 2, W + 2, ic]
        x = tf.pad(x, [[0, 0], [int(self.ph), int(self.ph)], [int(self.pw), int(self.pw)], [0, 0]])
        
        #[[B * H * W, ic]] * oc'
        map_00 = tf.gather_nd(x, grid_ix0iy0)
        map_01 = tf.gather_nd(x, grid_ix0iy1)
        map_11 = tf.gather_nd(x, grid_ix1iy1)
        map_10 = tf.gather_nd(x, grid_ix1iy0)

        #[[B,  H,  W, ic] * [B, H, W, 1] = [B, H, W, ic]] * oc'
        map_bilinear = (tf.expand_dims(w_11, axis = -1) * map_00 + \
                        tf.expand_dims(w_10, axis = -1) * map_01 + \
                        tf.expand_dims(w_00, axis = -1) * map_11 + \
                        tf.expand_dims(w_01, axis = -1) * map_10) * tf.expand_dims(mask, axis = -1)
        
        #[B, H, W, kh *kw * ic] #group always == 1
        map_all = tf.reshape(map_bilinear, [bs, ih, iw, -1])
        #[B, H, W, kh * kw * ic] convx [1, 1, kh * kw * ic, oc] = [B, H, W, oc]
        output = tf.nn.conv2d(map_all, tf.reshape(self.kernel, [1, 1, -1, self.filters]), strides = self.stride, padding = 'SAME')
        if self.use_bias:
            output += self.bias
        return output
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)
