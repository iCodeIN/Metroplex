import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2
from tensorflow.compat.v1 import nn
from functools import partial
from .helpers import get_1x1, get_3x3, DmolNet, draw_gaussian_diag_samples, \
gaussian_analytical_kl, layernorm, Attention, AttentionBlock, AttentionBlock2, FFN, CNN
from collections import defaultdict
import numpy as np
import itertools
from einops import rearrange, repeat


def positional_embedding(H):
    def inner(x, bsz):
        d_model = H.width // 2
        outer_res = H.data_scale // H.exp_scale
        pos_seq = tf.range(outer_res - 1, -1, -1.0)
        inv_freq = 1 / (10000 ** (tf.range(0, d_model, 2.0) / d_model))
        sinusoid_inp = tf.einsum('i,j->ij', inv_freq, pos_seq)
        x = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], axis=0)
        pos_emb = []
        pos_emb += [repeat(x, 'c h -> c h w', w=outer_res)]
        pos_emb += [repeat(x, 'c w -> c h w', h=outer_res)]
        pos_emb = tf.concat(pos_emb, axis=0)
        return tf.tile(pos_emb[None], [bsz, 1, 1, 1])
    return inner


class NullObj:
    def forward(self, x, **kwargs):
        return x

def null_op(x, **kwargs):
    return x

def upsample(size):
    def inner(x):
        return repeat(x, 'b c h w -> b c (h x) (w y)', x=size, y=size)
    return inner

class AttnEncBlock:
    def __init__(self, H, width, res, down_rate=None, init_scale=None, idx=None):
        self.H = H
        self.param_list = width, res, down_rate, init_scale, idx
        self.build()
    
    def build(self):
        H = self.H
        width, res, down_rate, init_scale, idx = self.param_list
        self.attn = AttentionBlock(self.H, width, res, init_scale=init_scale, name='enc-attn_'+str(idx))
        self.ffn = FFN(self.H, width=width, down_rate=down_rate, init_scale=init_scale)
        
    def forward(self, x):
        with tf.variable_scope("self-attention"):
            x = self.attn.forward(x)
        with tf.variable_scope("feedforward"):      
            x = self.ffn.forward(x)
        return x


class Block:
    def __init__(self, H, middle_width, out_width, down_rate=None, residual=False, use_3x3=True, zero_last=False, init_scale=None):
        self.H = H
        self.list = [middle_width, out_width, down_rate, use_3x3, zero_last, init_scale]
        self.residual = residual
        self.down_rate = down_rate
        self.build()
    
    def build(self):
        middle_width, out_width, down_rate, use_3x3, zero_last, init_scale = self.list
        H = self.H
        self.cs = []
        self.cs += [get_1x1(H, middle_width)]
        self.cs += [get_3x3(H, middle_width)] if use_3x3 else [get_1x1(H, middle_width)]
        self.cs += [get_3x3(H, middle_width)] if use_3x3 else [get_1x1(H, middle_width)]
        self.cs += [get_1x1(H, out_width, zero_weights=zero_last, init_scale=init_scale)]
        if self.down_rate is not None:
            if H.space2depth:
                self.down_sample = partial(tf.space_to_depth, self.down_rate, 'NCHW')
            else:
                self.down_sample = tf.keras.layers.AveragePooling2D(self.down_rate, strides=self.down_rate, padding='SAME', data_format='channels_first')
    
    def forward(self, x):
        xhat = x
        print(x.shape)
        for idx, conv in enumerate(self.cs):
            with tf.variable_scope("layer_" + str(idx)):
                xhat = conv(tf2.nn.gelu(xhat))
        out = x + xhat if self.residual else xhat
        print(xhat.shape)
        if self.down_rate is not None:
            out = self.down_sample(out)
        return out


def parse_layer_string(s):
    layers = []
    for ss in s.split(','):
        if 'x' in ss:
            res, num = ss.split('x')
            count = int(num)
            layers += [(int(res), None) for _ in range(count)]
        elif 'm' in ss:
            res, mixin = [int(a) for a in ss.split('m')]
            layers.append((res, mixin))
        elif 'd' in ss:
            res, down_rate = [int(a) for a in ss.split('d')]
            layers.append((res, down_rate))
        else:
            res = int(ss)
            layers.append((res, None))
    return layers


def pad_channels(t, width):
    w = width - t.shape[1] 
    return tf.pad(t, paddings = [[0, 0], [0, w], [0, 0], [0, 0]])
    

def get_width_settings(width, s):
    mapping = defaultdict(lambda: width)
    if s:
        s = s.split(',')
        for ss in s:
            k, v = ss.split(':')
            mapping[int(k)] = int(v)
    return mapping


def apply_and_checkpoint(f, params, **kwargs):
    if params.grad_checkpoint:
        @recompute_grad
        def inner(x, params):
            return f(**kwargs)
        return inner(**kwargs)
    else:
        return f(**kwargs)

class Encoder:
    def __init__(self, H):
        self.H = H
        self.build()

    def build(self):
        H = self.H
        self.widths = get_width_settings(H.outer_width, H.custom_width_str)
        self.in_conv = get_3x3(H, H.outer_width, name='in_conv') if self.H.exp_scale == 1 else get_1x1(H, H.outer_width, name='in_conv')
        enc_blocks = []
        blockstr = parse_layer_string(H.enc_blocks)
        n_blocks = len(blockstr)
        for idx, (res, down_rate) in enumerate(blockstr):
            use_3x3 = res > 2  # Don't use 3x3s for 1x1, 2x2 patches
            if self.H.transformer:
                enc_blocks.append(AttnEncBlock(H, self.widths[res], res, down_rate=down_rate, init_scale=n_blocks, idx=idx))
            else:
                enc_blocks.append(Block(H, int(self.widths[res] * H.bottleneck_multiple), self.widths[res], down_rate=down_rate, residual=True, use_3x3=use_3x3, init_scale=n_blocks))
        self.enc_blocks = enc_blocks
        if self.H.abs_enc:
            self.abs_enc = positional_embedding(H)

    def forward(self, x):
        x = tf.transpose(x, perm=[0, 3, 1, 2])
        x = rearrange(x, 'b c (h2 h1) (w2 w1) -> b (c h1 w1) h2 w2', h1=self.H.exp_scale, w1=self.H.exp_scale)
        x = self.in_conv(x)
        activations = {}
        if self.H.transformer:
            if self.H.abs_enc:
                x += self.abs_enc(x, tf.shape(x)[0])
        else:
            activations[int(x.shape[2])] = x
        for idx, block in enumerate(self.enc_blocks):
            with tf.variable_scope("block_" + str(idx)):
                #x = apply_and_checkpoint(block.forward, self.H, (x,))
                x = block.forward(x)
            res = int(x.shape[2])
            x = x if x.shape[1] == self.widths[res] else pad_channels(x, self.widths[res])
            if self.H.transformer:
                activations[idx] = x
            else:
                activations[res] = x
        return activations

class DecBlock:
    def __init__(self, H, res, mixin, n_blocks):
        self.H = H
        self.base = res
        self.mixin = mixin
        self.widths = get_width_settings(H.outer_width, H.custom_width_str)
        self.n_blocks = n_blocks
        self.build()
        
    def build(self):
        H = self.H
        res = self.base
        width = self.widths[res]
        use_3x3 = res > 2
        cond_width = int(width * H.bottleneck_multiple)
        self.zdim = width // H.zdim_factor
        self.enc = Block(H, cond_width, self.zdim * 2, residual=False, use_3x3=use_3x3)
        self.prior = Block(H, cond_width, self.zdim * 2 + width, residual=False, use_3x3=use_3x3, zero_last=True)
        self.z_fn = get_1x1(H, width, init_scale=self.n_blocks, name='z_fn')
        self.resnet = Block(H, cond_width, width, residual=True, use_3x3=use_3x3, init_scale=self.n_blocks)
        self.pre_sample = NullObj()
        self.normalize = null_op
        if self.mixin is not None:
            self.upsample = upsample(size=self.base // self.mixin)

    def _sample(self, x, res=None):
        with tf.variable_scope("prior"):        
            feats = self.prior.forward(x)
        pm, pv, xpp = feats[:, :self.zdim], feats[:, self.zdim:self.zdim * 2], feats[:, self.zdim * 2:]
        return xpp, pm, pv 
    
    def _enc(self, x, acts, res=None):
        return self.enc.forward(tf.concat([x, acts], axis=1))
    
    def sample(self, x, acts, res=None):
        with tf.variable_scope("enc"):
            qm, qv = tf.split(self._enc(x, acts, res=res), 2, axis=1)
        xpp, pm, pv = self._sample(x, res=res)
        z = draw_gaussian_diag_samples(qm, qv)
        kl = gaussian_analytical_kl(qm, pm, qv, pv)
        return z, xpp, kl

    def sample_uncond(self, x, t=None, lvs=None, res=None):
        n, c, h, w = x.shape.as_list()
        xpp, pm, pv = self._sample(x, res=res)        
        if lvs is not None:
            z = lvs
        else:
            if t is not None:
                pv = pv + tf.ones_like(pv) * np.log(t)
            z = draw_gaussian_diag_samples(pm, pv)
        return z, xpp

    def get_inputs(self, xs, activations):
        acts = activations[self.base]
        if self.base in xs:
            x = xs[self.base]
        else:
            x = tf.zeros_like(acts)
        x_shape = x.shape.as_list()
        x_shape[0] = tf.shape(acts)[0]
        x = tf.broadcast_to(x, x_shape)
        return x, acts

    def _forward(self, x, xs=None, acts=None, t=None, lvs=None, cond=False):
        if self.mixin is not None:
            with tf.variable_scope("upsample"):
                x += self.upsample(xs[self.mixin][:, :x.shape[1], ...])
        with tf.variable_scope("presample"):        
            x = self.pre_sample.forward(x)
        
        res = x
        x = self.normalize(x, axis=1, name="normalize")
        with tf.variable_scope("sample"):
            if cond:
                z, xpp, kl = self.sample(x, acts, res=res)
            else:
                z, xpp = self.sample_uncond(x, t, lvs=lvs, res=res)
        x = self.z_fn(z) + xpp + res
        with tf.variable_scope("resnet"):
            x = self.resnet.forward(x)
        if self.H.transformer:
            xs = x
        else:
            xs[self.base] = x
        if cond:
            return xs, dict(kl=kl)
        else:
            return xs

    def forward(self, xs, activations):
        x, acts = self.get_inputs(xs, activations)
        return self._forward(x, xs, acts=acts, cond=True)

    def forward_uncond(self, xs, t=None, lvs=None):
        if self.base in xs:
            x = xs[self.base]
        else:
            ref = xs[list(xs.keys())[0]]
            x = tf.zeros([tf.shape(ref)[0], self.widths[self.base], self.base, self.base], dtype=ref.dtype)
        self._forward(x, xs, t=t, lvs=lvs)

class AttnDecBlock(DecBlock):
    def __init__(self, H, res, mixin, n_blocks, idx):
        self.idx = idx
        super().__init__(H, res, mixin, n_blocks)
    
    def build(self):
        H = self.H
        res = self.base
        width = self.widths[res]
        self.zdim = width // H.zdim_factor
        self.z_fn = get_1x1(H, width, init_scale=self.n_blocks, name='z_fn')
        self.enc = AttentionBlock2(H, width, res, output_dim=self.zdim * 2, cross=True, name='dec-enc_'+str(self.idx), enc=True)
        self.normalize = layernorm
        if H.attn_block_opt == 0:
            self.pre_sample = AttentionBlock(H, width, res, init_scale=self.n_blocks, name='dec-pre_sample_'+str(self.idx))
            self.prior = get_1x1(H, self.zdim * 2, zero_weights=True)
        elif H.attn_block_opt == 1:
            self.pre_sample = NullObj()
            self.prior = AttentionBlock2(H, width, res, output_dim=self.zdim * 2, zero_weights=True, \
                          name='dec-prior_'+str(self.idx))
        self.resnet = FFN(H, width, init_scale=self.n_blocks)

    def get_inputs(self, xs, activations):
        idx = self.idx if self.H.enc_dec_mode else self.n_blocks - self.idx - 1
        acts = activations[idx]
        return xs, acts
    
    def _enc(self, x, acts, res=None):
        out, _ = self.enc.forward(x, acts, res=res)
        return out

    def _sample(self, x, res=None):
        if self.H.attn_block_opt == 0:
            with tf.variable_scope("prior"):        
                feats = self.prior(x)
            xpp = tf.zeros_like(x)
        elif self.H.attn_block_opt == 1:
            with tf.variable_scope("prior"):        
                feats, xpp = self.prior.forward(x, None, res=res)
        pm, pv = feats[:, :self.zdim], feats[:, self.zdim:self.zdim * 2]
        return xpp, pm, pv

    def forward_uncond(self, xs, t=None, lvs=None):
        self._forward(xs, xs=None, t=t, lvs=lvs)


class Decoder:
    def __init__(self, H):
        self.H = H
        self.build()

    def build(self):
        H = self.H
        resos = set()
        dec_blocks = []
        self.widths = get_width_settings(H.outer_width, H.custom_width_str)
        blocks = parse_layer_string(H.dec_blocks)
        for idx, (res, mixin) in enumerate(blocks):
            if H.transformer:
                block = AttnDecBlock(H, res, mixin, n_blocks=len(blocks), idx=idx)
            else:
                block = DecBlock(H, res, mixin, n_blocks=len(blocks))
            dec_blocks.append(block)
            resos.add(res)
        self.resolutions = sorted(resos)
        self.dec_blocks = dec_blocks
        zeros = tf.initializers.zeros(dtype=H.dtype)
        ones = tf.initializers.ones(dtype=H.dtype)
        if H.transformer:
            if self.H.abs_enc:
                self.abs_enc = positional_embedding(H)
            res = H.base_hidden_res
            name = 'bias_xs_' + str(res) 
            self.bias_xs = tf.get_variable(name, shape=(1, self.widths[res], res, res),
                                          initializer=zeros, trainable=True, dtype=H.dtype)
        else:
            self.bias_xs = []
            for res in self.resolutions: 
                if res <= H.no_bias_above:
                    name = 'bias_xs_' + str(res) 
                    self.bias_xs += [tf.get_variable(name, shape=(1, self.widths[res], res, res),
                                          initializer=zeros, trainable=True, dtype=H.dtype)]
        self.out_net = DmolNet(H)
        self.outer_res = res
        if H.final_fn:
            self.gain = tf.get_variable('gain', shape=(1, self.widths[res], 1, 1), initializer=ones, trainable=True, dtype=H.dtype)
            self.bias = tf.get_variable('bias', shape=(1, self.widths[res], 1, 1), initializer=zeros, trainable=True, dtype=H.dtype)
            self.final_fn = lambda x: x * self.gain + self.bias
        else:
            self.final_fn = null_op

    def forward(self, activations):
        stats = []
        if self.H.transformer:
            n = tf.shape(activations[0])[0]
            xs = tf.tile(self.bias_xs, multiples = [n, 1, 1, 1]) 
            if self.H.abs_enc:
                xs += self.abs_enc(xs, tf.shape(xs)[0])
        else: 
            xs = {int(a.shape[2]): a for a in self.bias_xs} 
        for idx, block in enumerate(self.dec_blocks):
            with tf.variable_scope("block_" + str(idx)):
                #xs, block_stats = apply_and_checkpoint(block.forward, self.H, (xs, activations))
                xs, block_stats = block.forward(xs, activations)
            stats.append(block_stats)
        x = xs if self.H.transformer else xs[self.outer_res]
        with tf.variable_scope("final_fn"):        
            x = self.final_fn(x)
        return x, stats

    def forward_uncond(self, n, t=None, y=None):
        if self.H.transformer:
            xs = tf.tile(self.bias_xs, multiples = [n, 1, 1, 1])
        else:
            xs = {}
            for bias in self.bias_xs:
                xs[int(bias.shape[2])] = tf.tile(bias, multiples = [n, 1, 1, 1])        
        for idx, block in enumerate(self.dec_blocks):
            with tf.variable_scope("block_" + str(idx)):
                xs = block.forward_uncond(xs, t)
        x = xs if self.H.transformer else xs[self.outer_res]
        with tf.variable_scope("final_fn"):        
            x = self.final_fn(x)
        return x


class Metroplex:
    def __init__(self, H):
        self.H = H
        self.build()

    def build(self):
        self.encoder = Encoder(self.H)
        self.decoder = Decoder(self.H)

    def forward(self, x):
        x_target = tf.identity(x)
        with tf.variable_scope("encoder"):
            activations = self.encoder.forward(x)
        with tf.variable_scope("decoder"):        
            px_z, stats = self.decoder.forward(activations)
            distortion_per_pixel, images = self.decoder.out_net.nll(px_z, x_target)
        rate_per_pixel = tf.zeros_like(distortion_per_pixel)
        ndims = tf.cast(tf.math.reduce_prod(x.shape[1:]), dtype=x.dtype)
        for statdict in stats:
            rate_per_pixel += tf.reduce_sum(statdict['kl'], axis=(1, 2, 3))
        rate_per_pixel /= ndims
        elbo = tf.reduce_mean(distortion_per_pixel + rate_per_pixel)
        return dict(elbo=elbo, distortion=tf.reduce_mean(distortion_per_pixel), rate=tf.reduce_mean(rate_per_pixel)), images

    def forward_uncond_samples(self, n_batch, t=None):
        with tf.variable_scope("decoder"):
            px_z = self.decoder.forward_uncond(n_batch, t=t)
            out = self.decoder.out_net.sample(px_z)
        return out
