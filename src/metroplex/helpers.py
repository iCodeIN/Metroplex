import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import nn
import tensorflow.compat.v2 as tf2
from functools import partial
import numpy as np
from einops import rearrange, repeat

def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
    return -0.5 + logsigma2 - logsigma1 + 0.5 * (tf.math.exp(logsigma1) ** 2 + (mu1 - mu2) ** 2) / (tf.math.exp(logsigma2) ** 2)


def draw_gaussian_diag_samples(mu, logsigma):
    eps= tf.random.normal(tf.shape(mu), dtype=mu.dtype)
    return tf.exp(logsigma) * eps + mu


def get_conv(H, out_dim, kernel_size, zero_weights=False, init_scale=None, name=None):
    name = 'conv' if name is None else name 
    if zero_weights and H.zero_weights:
        initializer = tf.initializers.zeros(dtype=H.dtype)
    else:
        if H.init_std:
            stddev = H.init_std
        else:
            stddev = np.sqrt(1 / out_dim)
        if H.init_scale is not None:
            init_scale = H.init_scale
        if init_scale is not None:
            stddev *= np.sqrt(1 / init_scale) 
        initializer = tf.random_uniform_initializer(minval=-stddev, maxval=stddev, dtype=H.dtype)
    return tf.keras.layers.Conv2D(out_dim, kernel_size, padding='same', kernel_initializer=initializer, data_format='channels_first', name=name)


def get_3x3(H, out_dim, groups=1, init_scale=None, name=None):
    return get_conv(H, out_dim, 3, init_scale=init_scale, name=name)


def get_1x1(H, out_dim, groups=1, zero_weights=False, init_scale=None, name=None):
    return get_conv(H, out_dim, 1, zero_weights=zero_weights, init_scale=init_scale, name=name)

def layernorm(x, name="layer_norm", axis=None, epsilon=1e-5):
    return x

'''def layernorm(x, name="layer_norm", axis=None, epsilon=1e-5):
    lnorm = tf.keras.layers.LayerNormalization(axis=axis, epsilon=epsilon)
    return lnorm(x)'''

def postprocess(H, width):
    def inner(x):
        if H.rezero:
            initializer = tf.initializers.zeros(dtype=H.dtype)
            rezero = tf.get_variable("rezero_bias", [], initializer=initializer, dtype=H.dtype)
            return x * rezero
        else:
            return x
    return inner

def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    m = tf.math.reduce_max(x, axis=-1, keepdims=True)
    return x - m - tf.math.log(tf.reduce_sum(tf.exp(x - m), axis=-1, keepdims=True))


def const_max(t, constant):
    other = tf.ones_like(t) * constant
    return tf.math.maximum(t, other)


def const_min(t, constant):
    other = tf.ones_like(t) * constant
    return tf.math.minimum(t, other)


class DmolNet:
    def __init__(self, H):
        self.H = H
        self.num_params_per_mixture = 10
        '''if H.shared_sigma: # no std
            self.num_params_per_mixture -= 3
        if H.color_non_ar: # no coeff
            self.num_params_per_mixture -= 3
        if H.num_mixtures == 1: # no logit_probs
            self.num_params_per_mixture -= 1'''  
        self.build()          

    def build(self):
        out_dim = self.H.num_mixtures * self.num_params_per_mixture * (self.H.exp_scale ** 2)
        self.out_conv = get_conv(self.H, out_dim, kernel_size=1, name='out_conv')
        # TODO!!!!!!!! check how to make this stable
        if self.H.shared_sigma:
            self.shared_sigma = tf.get_variable('sigma', shape=(1,), initializer='ones', dtype=H.dtype, trainable=True)

    def nll(self, px_z, x):
        return self.discretized_mix_logistic_loss(x=x, l=self.forward(px_z))

    def forward(self, px_z):
        xhat = self.out_conv(px_z)
        if self.H.exp_scale > 1:
            xhat = rearrange(xhat, 'b (c h1 w1) h2 w2 -> b c (h2 h1) (w2 w1)', h1=self.H.exp_scale, w1=self.H.exp_scale)
        return tf.transpose(xhat, perm=[0, 2, 3, 1])

    def sample(self, px_z):
        im = self.sample_from_discretized_mix_logistic(self.forward(px_z), self.H.num_mixtures)
        xhat = (im + 1.0) * 127.5
        #TODO: deal with this
        xhat = xhat.detach().cpu().numpy()
        xhat = np.minimum(np.maximum(0.0, xhat), 255.0).astype(np.uint8)
        return xhat

    def discretized_mix_logistic_loss(self, x, l):
        """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
        # Adapted from https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py
        xs = [s for s in x.shape]  # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
        ls = [s for s in l.shape]  # predicted distribution, e.g. (B,32,32,100)
        xs[0] = ls[0] = tf.shape(x)[0]
        nr_mix = self.H.num_mixtures  # here and below: unpacking the params of the mixture of logistics
        if self.H.num_mixtures != 1:
            logit_probs = l[:, :, :, :nr_mix]
        l = tf.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
        means = l[:, :, :, :, :nr_mix]
        log_scales = const_max(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
        if self.H.shared_sigma:
            log_scales = const_max(tf.fill(tf.shape(l[:, :, :, :, nr_mix:2 * nr_mix]), self.shared_sigma), -7.)            
        coeffs = tf.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
        if self.H.color_non_ar:
            coeffs = tf.zeros_like(coeffs)
        x = tf.reshape(x, xs + [1]) + tf.zeros(xs + [nr_mix], dtype=x.dtype)  # here and below: getting the means and adjusting them based on preceding sub-pixels
        m2 = tf.reshape(means[:, :, :, 1, :] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :], [xs[0], xs[1], xs[2], 1, nr_mix])
        m3 = tf.reshape(means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :], [xs[0], xs[1], xs[2], 1, nr_mix])
        means = tf.concat([tf.reshape(means[:, :, :, 0, :], [xs[0], xs[1], xs[2], 1, nr_mix]), m2, m3], axis=3)
        
        centered_x = x - means
        inv_stdv = tf.exp(-log_scales)
        plus_in = inv_stdv * (centered_x + 1. / 255.)
        cdf_plus = tf.sigmoid(plus_in)
        min_in = inv_stdv * (centered_x - 1. / 255.)
        cdf_min = tf.sigmoid(min_in)
        log_cdf_plus = plus_in - tf.math.softplus(plus_in)  # log probability for edge case of 0 (before scaling)
        log_one_minus_cdf_min = -tf.math.softplus(min_in)  # log probability for edge case of 255 (before scaling)
        cdf_delta = cdf_plus - cdf_min  # probability for all other cases
        mid_in = inv_stdv * centered_x
        log_pdf_mid = mid_in - log_scales - 2. * tf.math.softplus(mid_in)  # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)

        # now select the right output: left edge case, right edge case, normal case, extremely low prob case (doesn't actually happen for us)

        # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
        # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.math.log(cdf_delta)))

        # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
        # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
        # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
        # if the probability on a sub-pixel is below 1e-5, we use an approximation based on the assumption that the log-density is constant in the bin of the observed sub-pixel value
        log_probs = tf.where(x < -0.999,
                                log_cdf_plus,
                                tf.where(x > 0.999,
                                            log_one_minus_cdf_min,
                                            tf.where(cdf_delta > 1e-5,
                                                        tf.math.log(const_max(cdf_delta, 1e-12)),
                                                        log_pdf_mid - np.log(127.5))))
        log_probs = tf.reduce_sum(log_probs, axis=3)
        if self.H.num_mixtures != 1:
            log_probs += log_prob_from_logits(logit_probs)
        mixture_probs = tf.reduce_logsumexp(log_probs, axis=-1)
        output = -1. * tf.reduce_sum(mixture_probs, axis=[1, 2]) / np.prod([int(x) for x in xs[1:]])
        return output, means


    def sample_from_discretized_mix_logistic(self, l):
        ls = [s for s in l.shape.as_list()]
        ls[0] = tf.shape(l)[0]
        xs = ls[:-1] + [3]
        # unpack parameters
        if self.H.num_mixtures != 1:
            logit_probs = l[:, :, :, :nr_mix]
            l = tf.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
        else:
            l = tf.reshape(l, xs + [nr_mix * 3])
        if self.H.num_mixtures != 1:
            # sample mixture indicator from softmax
            eps = tf.random.uniform(tf.shape(logit_probs), 1e-5, 1. - 1e-5, dtype=logit_probs.dtype)
            amax = tf.math.argmax(logit_probs - tf.math.log(-tf.math.log(eps)), axis=3)
            sel = tf.one_hot(amax, depth=nr_mix, axis=-1, dtype=l.dtype)
            sel = tf.reshape(sel, xs[:-1] + [1, nr_mix])
        else:
            sel = tf.ones(xs[:-1] + [1, nr_mix], dtype=l.dtype)
        # select logistic parameters
        means = tf.reduce_sum(l[:, :, :, :, :nr_mix] * sel, axis=4)
        log_scales = const_max(tf.reduce_sum(l[:, :, :, :, nr_mix:nr_mix * 2] * sel, axis=4), -7.)
        if self.H.shared_sigma:
            #TODO: fix here!!!!
            log_scales = const_max(tf.fill(tf.shape(l), self.shared_sigma), -7.)            
        coeffs = tf.reduce_sum(tf.tanh(l[:, :, :, :, nr_mix * 2:nr_mix * 3]) * sel, axis=4)
        if self.H.color_non_ar:
            coeffs = tf.zeros_like(coeffs) 
        # sample from logistic & clip to interval
        # we don't actually round to the nearest 8bit value when sampling
        u = tf.random.uniform(tf.shape(means), 1e-5, 1. - 1e-5, dtype=means.dtype)
        x = means + tf.exp(log_scales) * (tf.math.log(u) - tf.math.log(1. - u))
        x0 = const_min(const_max(x[:, :, :, 0], -1.), 1.)
        x1 = const_min(const_max(x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, -1.), 1.)
        x2 = const_min(const_max(x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, -1.), 1.)
        return tf.concat([tf.reshape(x0, xs[:-1] + [1]), tf.reshape(x1, xs[:-1] + [1]), tf.reshape(x2, xs[:-1] + [1])], axis=3)

class CNN:
    def __init__(self, H, dim, cur_res, output_dim=None, cross=False, init_scale=None, name=None):
        self.H = H
        self.param_list = [dim, output_dim, cross, init_scale]
        self.build()

    def build(self):
        dim, out_width, cross, init_scale = self.param_list
        H = self.H
        out_width = out_width if out_width else dim
        self.cross = cross
        zero_last = False
        middle_width = dim // 4
        
        self.cs = []
        self.cs += [get_1x1(H, middle_width)]
        self.cs += [get_3x3(H, middle_width)]
        self.cs += [get_3x3(H, middle_width)]
        self.cs += [get_1x1(H, out_width, zero_weights=zero_last, init_scale=init_scale)]

    def forward(self, xhat, ctx=None):
        if self.cross:
            xhat = tf.concat([xhat, ctx], axis=1)
        for idx, conv in enumerate(self.cs):
            with tf.variable_scope("layer_" + str(idx)):
                xhat = conv(tf2.nn.gelu(xhat))
        return xhat

'''class Attention:
    def __init__(self, H, dim, cur_res, output_dim=None, cross=False, init_scale=None, name=None):
        self.H = H
        self.local = H.local_attn_scale < cur_res
        self.param_list = [dim, name, output_dim, cross]
        self.attn_scale = self.H.local_attn_scale
        self.build()

    def build(self):
        H = self.H
        dim, name, output_dim, self.cross = self.param_list
        qkv_dim = dim if self.cross else 3 * dim
        self.qkv = get_1x1(H, qkv_dim, name='qkv')
        out_dim = output_dim if output_dim else dim 
        self.out = get_1x1(H, out_dim, name='out')
        self.length = H.local_attn_scale ** 2
        self.num_heads = dim // H.head_dim
        self.postprocess = postprocess(H, dim)
        stddev = np.sqrt(1 / dim) 
        initializer = tf.random_uniform_initializer(minval=-stddev, maxval=stddev, dtype=H.dtype)
        #self.rel_bias = tf.get_variable(name + "-rel_bias", [self.num_heads, self.length, 2*H.head_dim], initializer=initializer, dtype=H.dtype)
        #self.rel_bias = tf.pad(self.rel_bias, tf.constant([[0, 0], [0, 0], [0, H.head_dim]]))        
        #self.rel_bias = tf.get_variable(name + "-rel_bias", [self.num_heads, 2 * self.length - 1], initializer=initializer, dtype=H.dtype)
              
        self.attn_options = self.attn_opt() 
        if self.local:
            self.shift = np.zeros((4, 2))
            self.pad_units = H.local_attn_scale // 2
            self.shift[1:-1] = np.ones(2, 2) * self.pad_units
            self.shift = tf.constant(self.shift.astype('int32'))

    def attn_opt(self):
        # len = num_heads; 0: normal 2D local, 1: shifted 2D local, 2: vertical, 3: horizontal
        attn_opt = self.H.attn_options
        if not isinstance(attn_opt, list): 
            if attn_opt == 0:
                return [0, 1] * (self.num_heads // 2)
            elif attn_opt == 1:
                return [1, 2] * (self.num_heads // 2)
            elif attn_opt == 2:
                return [0, 1, 2, 3] * (self.num_heads // 4) + [0, 1] * (self.num_heads % 4)

    def fold(self, x):
        x = rearrange(x, 'b (head_types num_heads head_dim) h w -> b num_heads h w (head_types head_dim)', num_heads=self.num_heads)
        num_heads = int(tf.shape(x)[1])
        resolution = int(x.shape[2] * x.shape[3])
        xs = tf.split(x, num_heads, axis=1)
        biases = tf.split(self.rel_bias, num_heads, axis=1)
        out = []
        self.bs_per_head = []
        for idx, attn_opt in enumerate(self.attn_options):
            if attn_opt in [0, 1]:
                if attn_opt == 1:
                    x = tf.pad(x, self.shift)
                x = rearrange(x, 'b (h1 h2) (w1 w2) c -> (h1 w1 b) (h2 w2) c', h2=self.attn_scale, w2=self.attn_scale)
            else:
                if attn_opt == 2:
                    x = rearrange(x, 'b h w c -> b w h c')
                x = rearrange(x, 'b (h1 h2) w c -> (h1 b) (h2 w) c', h1=resolution//self.length)
            self.bs_per_head += [int(tf.shape(x)[0])]
            out += [x + biases[idx]]
        return tf.split(tf.concat(out), 3, axis=-1)

    def refold(self, x, batch_size):
        xs = tf.split(x, self.bs_per_head, axis=0)
        out = []
        for idx, attn_opt in enumerate(self.attn_options):
            x = xs[idx]
            extra_bs = self.bs_per_head[idx] // batch_size
            if attn_opt in [0, 1]:
                h1 = w1 = int(np.sqrt(extra_bs))
                x = rearrange(x, '(h1 w1 b) h2 w2 c -> b c (h1 h2) (w1 w2)', h1=h1, w1=w1)
                if attn_opt == 1:
                    x = x[..., self.pad_units:-self.pad_units, self.pad_units:-self.pad_units]
            else:
                x = rearrange(x, '(h1 b) h2 w c -> b c (h1 h2) w', h1=extra_bs)
                if attn_opt == 2:
                    x = rearrange(x, 'b c h w -> b c w h')
            out += [x]
        return tf.concat(out, axis=1)
            

    def forward(self, inp, ctx=None):
        attn_scale = self.attn_scale
        if self.cross:
            k = v = ctx
            q = self.qkv(inp)
            qkv = tf.concat([q, k, v], axis=1)
        else:
            qkv = self.qkv(inp)
        #print('1', qkv.shape)
                        
        # pad and fold the tensors
        if self.local:
            batch_size = int(tf.shape(x)[0])
            q, k, v = self.fold(x)
        else:
            x = rearrange(qkv, 'b (head_types num_heads head_dim) h w -> b num_heads (h w) (head_types head_dim)', head_types=3, num_heads=self.num_heads)
            # add local relative positional encoding
            #x += self.rel_bias
            x = rearrange(x, 'b num_heads length dim -> (num_heads b) length dim')
            q, k, v = tf.split(x, 3, axis=-1)
        #print('4', q.shape, k.shape)
        qk = tf.einsum('bqd, bkd -> bqk', q, k) * (int(q.shape[-1]) ** (-0.5))
        #qk += tf.broadcast_to(self.rel_bias, tf.shape(qk))
        qk_soft = nn.softmax(qk, axis=-1)
        x = tf.einsum('bqk, bkd -> bqd', qk_soft, v)
        #print('6', x.shape)
        # unfold the tensor
        if self.local:
            x = self.refold(x, batch_size)
        else:
            x = rearrange(x, '(num_heads b) (h w) head_dim -> b (num_heads head_dim) h w', num_heads=self.num_heads, h=attn_scale)
        #print('7', x.shape)
        x = self.out(x)
        return self.postprocess(x)'''

class Attention:
    def __init__(self, H, dim, cur_res, output_dim=None, cross=False, init_scale=None, name=None):
        self.H = H
        self.param_list = [dim, name, output_dim, cross]
        self.attn_scale = self.H.local_attn_scale
        self.build()

    def build(self):
        H = self.H
        dim, name, output_dim, self.cross = self.param_list
        qkv_dim = dim if self.cross else 3 * dim
        self.qkv = get_1x1(H, qkv_dim, name='qkv')
        out_dim = output_dim if output_dim else dim 
        self.out = get_1x1(H, out_dim, name='out')
        length = self.attn_scale
        self.num_heads = dim // H.head_dim
        self.postprocess = postprocess(H, dim)
        def init(shape, dtype):
            x = np.zeros((2*length-1,))
            x[:length] = np.linspace(0, 1, num=length)
            x[length:] = x[length-2::-1]
            x = repeat(x, 'length -> num_heads length', num_heads = self.num_heads)
            return tf.constant(x, dtype=H.dtype)
        def f(idx):
            x = tf.get_variable(name + "-rel_bias"+str(idx), [self.num_heads, 2*length-1], initializer=init, dtype=H.dtype)
            x = repeat(x, 'n k -> n q k', q = length)
            x_size = tf.shape(x)
            x = tf.pad(x, [[0, 0], [0, 0], [1, 0]])
            x = tf.reshape(x, [x_size[0], x_size[2] + 1, x_size[1]])
            x = tf.slice(x, [0, 1, 0], [-1, -1, -1])
            x = tf.reshape(x, x_size)[..., :length]
            return x

        x1, x2 = map(f, (1, 2))
        self.rel_bias = tf.einsum('nab, ncd -> nacbd', x1, x2)

    def forward(self, inp, ctx=None):
        attn_scale = self.attn_scale
        if self.cross:
            k = v = ctx
            q = self.qkv(inp)
        else:
            q, k, v = tf.split(self.qkv(inp), 3, axis=1)
        q, k, v = map(lambda x: rearrange(x, 'b (num_heads head_dim) h w -> b num_heads (h w) head_dim', num_heads=self.num_heads), (q, k, v)) 
        qk = tf.einsum('bnqd, bnkd -> bnqk', q, k) 
        qk = rearrange(qk, 'b num_heads (h1 w1) (h2 w2) -> b num_heads h1 w1 h2 w2', h1=attn_scale, h2=attn_scale)
        bias = tf.tile(self.rel_bias[None], multiples = [tf.shape(qk)[0], 1, 1, 1, 1, 1])
        norm_q = repeat(tf.norm(tf.stop_gradient(q), axis=-1), 'b num_heads (h1 w1) -> b num_heads h1 w1 h2 w2', h1=attn_scale, h2=attn_scale, w2=attn_scale)
        qk += norm_q * bias
        qk = rearrange(qk, 'b num_heads h1 w1 h2 w2 -> b num_heads (h1 w1) (h2 w2)')
        qk *= (self.H.head_dim ** (-0.5))
        qk_soft = nn.softmax(qk, axis=-1)
        x = tf.einsum('bnqk, bnkd -> bnqd', qk_soft, v)
        x = rearrange(x, 'b num_heads (h w) head_dim -> b (num_heads head_dim) h w', h=attn_scale)
        x = self.out(x)
        return self.postprocess(x)


class AttentionBlock:
    def __init__(self, H, dim, cur_res, init_scale=None, name=None):
        self.param_list = H, dim, cur_res, init_scale, name
        self.build()

    def build(self):
        H, dim, cur_res, init_scale, name = self.param_list
        self.attention = Attention(H, dim, cur_res, init_scale=init_scale, name=name)
        self.layernorm = layernorm

    def forward(self, x):
        res = x
        x = self.layernorm(x, axis=1)
        x = self.attention.forward(x)
        return x + res

class AttentionBlock2:
    def __init__(self, H, dim, cur_res, output_dim, cross=False, zero_weights=False, name=None, enc=False):
        self.param_list = H, dim, cur_res, output_dim, cross, zero_weights, name, enc
        self.build()

    def build(self):
        H, dim, cur_res, output_dim, cross, zero_weights, name, enc = self.param_list
        use_cnn = enc and H.cnn_enc
        if use_cnn:
            self.attention = CNN(H, dim, cur_res, cross=cross, name=name)
        else:
            self.attention = Attention(H, dim, cur_res, cross=cross, name=name)
        self.layernorm = layernorm
        self.linear = get_1x1(H, output_dim, zero_weights=zero_weights, name='proj_z')

    def forward(self, x, acts, res):
        x = self.attention.forward(x, ctx=acts)
        out = self.linear(self.layernorm(x+res, axis=1))
        return out, x

class FFN:
    def __init__(self, H, width, down_rate=None, init_scale=None):
        self.H = H
        self.width = width
        self.init_scale = init_scale
        self.down_rate = down_rate
        self.build()
    
    def build(self):
        H = self.H
        self.cs = []
        self.cs += [get_1x1(H, 4 * self.width)]
        self.cs += [get_1x1(H, self.width, init_scale=self.init_scale)]
        self.layernorm = layernorm
        self.postprocess = postprocess(H, self.width)

    def forward(self, x):
        xhat = x
        xhat = self.layernorm(xhat, axis=1)
        for idx, conv in enumerate(self.cs):
            with tf.variable_scope("layer_" + str(idx)):
                xhat = conv(xhat)
                if idx == 0:
                    xhat = tf2.nn.gelu(xhat)
        out = x + self.postprocess(xhat)
        if self.down_rate is not None:
            out = self.down_sample(out)
        return out