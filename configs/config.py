import tensorflow.compat.v1 as tf
import numpy as np
from functools import partial 
import json
import math 

# this allows both dict and class expression for convenience
class Hyperparams(dict):
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return None

    def __setattr__(self, attr, value):
        self[attr] = value

def print_params(**kwargs):
    print(kwargs)

def prog_config(params):
    if params.prog_mode is not False:
        if params.prog_mode == 0:
            if params.data_scale == 32:
                assert params.width == 1024
                assert params.depth == 12
                params.enc_blocks = '32x11,32d2,16x5,16d2,8x1,8d2,4x1,4d4,1x1'
                params.dec_blocks = "1x1,4m1,8m4,8x1,16m8,16x1,32m16,32x5"
                params.custom_width_str = "32:512,16:1024,8:2048,4:2048,1:2048"
                params.outer_width = 512
            elif params.width == 768:
                if params.depth == 24:
                    params.enc_blocks = '64x23,64d2,32x11,32d2,16x5,16d2,8x1,8d2,4x1,4d4,1x2'
                    params.dec_blocks = "1x1,4m1,8m4,16m8,16x2,32m16,32x5,64m32,64x11"
                elif params.depth == 12:
                    params.enc_blocks = '64x11,64d2,32x5,32d2,16x1,16d2,8x1,8d2,4x1,4d4,1x2'
                    params.dec_blocks = "1x1,4m1,8m4,16m8,32m16,32x2,64m32,64x5"
                params.custom_width_str = "64:192,32:384,16:768,8:1536,4:3072,1:3072"
                params.outer_width = 192
            elif params.width == 1024:
                if params.depth == 24:
                    params.enc_blocks = '128x23,128d2,64x11,64d2,32x5,32d2,16x1,16d2,8x1,8d2,4x1,4d4,1x2'
                    params.dec_blocks = "1x1,4m1,8m4,16m8,32m16,32x2,64m32,64x5,128m64,128x11"
                elif params.depth == 12:
                    params.enc_blocks = '128x11,128d2,64x5,64d2,32x1,32d2,16x1,16d2,8x1,8d2,4x1,4d4,1x2'
                    params.dec_blocks = "1x1,4m1,8m4,16m8,32m16,64m32,64x2,128m64,128x5"
                params.custom_width_str = "128:128,64:256,32:512,16:1024,8:2048,4:4096,1:4096"
                params.outer_width = 128
        elif params.prog_mode == 1:
            params.custom_width_str = ''
            if params.depth == 24:
                if params.start_res == 16:
                    params.enc_blocks = '16x11,16d2,8x11,8d2,4x11,4d4,1x11'
                    params.dec_blocks = "1x5,4m1,4x5,8m4,8x5,16m8,16x5"
                    params.outer_width = params.width
                elif params.start_res == 32:
                    params.enc_blocks = '32x9,32d2,16x9,16d2,8x9,8d2,4x9,4d4,1x9'
                    params.dec_blocks = "1x4,4m1,4x4,8m4,8x4,16m8,16x4,32m16,32x4"
                    params.outer_width = params.width // 2
            elif params.depth == 12:
                if params.start_res == 16:
                    params.enc_blocks = '16x5,16d2,8x5,8d2,4x5,4d4,1x5'
                    params.dec_blocks = "1x2,4m1,4x2,8m4,8x2,16m8,16x2"
                    params.outer_width = params.width
                elif params.start_res == 32:
                    params.enc_blocks = '32x5,32d2,16x5,16d2,8x3,8d2,4x3,4d4,1x3'
                    params.dec_blocks = "1x1,4m1,4x1,8m4,8x1,16m8,16x2,32m16,32x2"
                    params.outer_width = params.width // 2
        largest_res = int(params.enc_blocks.split('x')[0])
        params.exp_scale = params.data_scale // largest_res
        #print_params(exp_scale = params.exp_scale, 
        #      dec_blocks = params.dec_blocks, enc_blocks = params.enc_blocks, custom_width_str = params.custom_width_str)
    else:
        params.outer_width = params.width
        params.custom_width_str = ''
        params.enc_blocks = params.dec_blocks = str(params.base_hidden_res) + "x" + str(params.depth)
        params.exp_scale = params.data_scale // params.base_hidden_res
    return params


class Hparams(Hyperparams):
    def __init__(self):
        super().__init__(self)
        # you can add parameters from json iff they are listed here as self.(variable name) 
        # if you want to add more parameters, you need to register them 
        # here by setting self.(your parameter name) = (some default value)
        # the list of hparams here is aimed for maximal coverage, not meant to be exhaustive
        # CONSTRAINED: the parameter is determined by other json parameters and cannot be 
        # changed directly. added here for reference.
        # OPTIONAL: json file doesn't necessarily need to specify the parameter. 
        # MANDATORY: json file needs to specify the parameter.
        CONSTRAINED = OPTIONAL = MANDATORY = None

        # optimization------------------------
        # maybe changed
        self.gradient_clipping = 200 
        self.lr = 0.0003

        # fixed
        self.optimizer = "adam"
        self.lr_decay = "cosine"
        self.warmup_steps = 100
        self.skip_threshold = 300
        self.beta_1 = 0.9
        self.beta_2 = 0.999

        # training misc.---------------------
        # maybe changed
        self.dataset = MANDATORY
        self.steps_per_checkpoint = 1000 
        self.iterations = 500
        self.iters_per_host_call = 100
        self.grad_checkpoint = False 
        self.train_batch_size = 256 # eval/pred batch_size is made equal to this by default
        self.eval_batch_size = OPTIONAL
        self.predict_batch_size = OPTIONAL
        self.train_steps = 40000
        self.predict_steps = 0
        self.ema_rate = 0 # not implemented
        
        # one of the two below is mandatory 
        self.eval_dataset_size = OPTIONAL
        self.eval_steps = OPTIONAL

        self.batch_size_inv_factor = OPTIONAL
        self.iter_factor = OPTIONAL

        # fixed
        self.use_bf16 = False # substantial perf improvement from this being false w/ negligible speed-down and memory increase
        self.seed = 0
        self.n_channels = 3
        
        # model------------------------------
        # maybe changed
        self.width = 384
        self.depth = 24
        self.base_hidden_res = 16
        self.prog_mode = False
        self.start_res = OPTIONAL
        self.outer_width = CONSTRAINED
        self.zdim_factor = 32
        self.zero_weights = True
        self.final_fn = False
        self.model_path = MANDATORY
        self.custom_width_str = CONSTRAINED
        self.enc_blocks = self.dec_blocks = CONSTRAINED
        self.exp_scale = CONSTRAINED

        # fixed
        self.zdim = CONSTRAINED
        self.no_bias_above = 64
        self.space2depth = False
        self.bottleneck_multiple = 0.25

        # DMOL specific-----------------------
        # maybe changed
        self.num_mixtures = 10
        self.shared_sigma = None
        self.color_non_ar = True

        # transformer specific----------------
        # maybe changed
        self.transformer = True

        self.attn_options = 0
        # option to use which sparse attention. likely to be not used and therefore not tested yet. 
        # 0: normal 2D local, 1: shifted 2D local, 2: vertical, 3: horizontal
        
        self.attn_block_opt = 0
        # 0: self-attn -> (linear, cross-attn) -> FFN
        # 1: (self-attn, cross-attn) -> FFN

        self.enc_dec_mode = True
        # True: enc-dec-like architecture
        # False: UNet-like architecture

        # fixed
        self.local_attn_scale = CONSTRAINED     
        self.head_dim = 128 

        # tempory params for exps
        self.rezero = True
        self.abs_enc = False
        self.init_std = None
        self.init_scale = None
        self.enc_mode = 'attn'
        self.random_init_posenc = False

        #self.save_freq = 10
        #self.thresholding = True

    def check_input(self, input):
        valid_keys = self.keys()
        for key in input.keys():
            assert key in valid_keys
    
    def update(self, input):
        self.check_input(input)
        super().update(input)

        self.zdim = self.width // self.zdim_factor

        self.local_attn_scale = self.base_hidden_res # needs fix if this is too big  

        for mode in ['eval', 'predict']:
            if self[mode + '_batch_size'] is None:
                self[mode + '_batch_size'] = self.train_batch_size

        if self.batch_size_inv_factor:
            for mode in ['train', 'eval']:
                self[mode + '_batch_size'] = int(self[mode + '_batch_size'] / self.batch_size_inv_factor)

        if self.iter_factor:
            self.train_steps = int(self.train_steps * self.iter_factor) 
            
        if self.eval_steps is None:
            print('eval_steps is not None. Evaluation will be on the whole evaluation'\
                  'dataset. If you want to skip evaluation, set it to 0 instead.')
            assert self.eval_dataset_size
            self.eval_steps = self.eval_dataset_size // self.eval_batch_size

        if self.transformer:
            assert self.prog_mode is False

    def produce_hparams(self, json_file):
        print(json_file)
        with open(json_file) as f:
            input = json.load(f)
        self.update(input)
        self.data_scale = self.dataset["image_size"]
        self.dtype = tf.bfloat16 if self.use_bf16 else tf.float32
        return prog_config(self)


def retrieve_hparams(json_file):
    params = Hparams()
    params.produce_hparams(json_file)
    print(dict(params))
    return dict(params)