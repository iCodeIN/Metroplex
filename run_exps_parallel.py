import subprocess
import os
from config_maker import make_configs, split_list
import argparse
from make_tpu import TPUMaker
import logging
from copy import deepcopy
from functools import partial

def pre_marge(x, y):
    new_dict = deepcopy(x)
    new_dict.update(**x, **y)
    return new_dict

_marge = lambda x: partial(pre_marge, x)

exps = dict()

# WARNING: don't forget to set the best config after each experiment.

# exp 1: conv vs. transformer---------------
# compare in terms of FLOPS, memory and throughput

marge = _marge({})

# conv prog_mode = 0
exps["exp-1-a"] = marge({'width': [768, 1024], 'transformer': False, 'prog_mode': 0})
exps["exp-1-b"] = marge({'width': 768, 'depth': 12, 'transformer': False, 'prog_mode': 0})

# conv prog_mode = False (ablation)
exps["exp-1-e"] = marge({'width': [768, 1024], 'transformer': False})
exps["exp-1-f"] = marge({'width': 768, 'depth': 12, 'transformer': False})

# trans
exps["exp-1-g"] = marge({'width': [384, 512]})
exps["exp-1-h"] = marge({'width': 384, 'depth': 12})

exps["exp-1-i"] = marge({'attn_block_opt': [0, 1], 'enc_dec_mode': [True, False]})
exps["exp-1-j"] = marge({'enc_mode': ['cnn', 'ffn']})
# good_config = {...}
# add this to the next config

# exp 2: measure variance
# exps["exp-2-a"] = {'width': [512, 512, 512, 512, 512]}

# exp 3: lr & clip_grad & zero_weights & lr_decay, init_scale
marge = _marge({}) # fill this
exps["exp-3-a"] = marge({'lr': [0.0001, 0.003, 0.001], 'gradient_clipping': [1.0, 200.0], 'lr_decay': ["none", "cosine"]}
exps["exp-3-b"] = marge({'zero_weights': False}

# exp 4: batch size: 
marge = _marge({}) # fill this
exps["exp-4-a"] = marge({'batch_size_inv_factor': 4, 'iter_factor': 4}) 
# use v3-128/256
exps["exp-4-b"] = marge({'batch_size_inv_factor': 1/4, 'iter_factor': 1/4, 'lr': [...]}) # try an increased lr 
#exps["exp-3-c"] = {'final_fn': False}

# exp 5: vary width, depth, base_hidden_res
marge = _marge({}) # fill this
exps["exp-5-a"] = marge({'width': [256, 384], 'zdim_factor': [16, 32, 64]}) # if error, try a different lr
# pick the best zdim_factor below
exps["exp-5-b"] = marge({'base_hidden_res': 8})
exps["exp-5-c"] = marge({'depth': 12})

# increase the dim (most likely depth) steepest and decrease the dim least steep
# repeat the same exp again below (note that some configs were already done above)

exps["exp-5-d"] = marge({'width': ..., 'zdim_factor': ...}) # pick the best zdim_factor
exps["exp-5-e"] = marge({'base_hidden_res': ...})
exps["exp-5-f"] = marge({'depth': ...})

# when you get to the local optimum, double/halve the computes 

# exp 6: DMOL
marge = _marge({}) # fill this
exps["exp-6-a"] = marge({'num_mixtures': [1, 4, 40]})

# exp 7: misc.
#exps["exp-7-a"] = {'abs_enc': 'per_layer'} # ablation
exps["exp-7-b"] = marge({'random_init_posenc': True}) # ablation
#exps["exp-7-c"] = {'ema_rate': 0.999} # ablation, not implemented and optional

# exp 8: train with optimal scaling for many iters to find the intersection  
marge = _marge({}) # fill this
exps["exp-8-a"] = marge({'width': ..., 'depth': ..., 'base_hidden_res', ...}) #larger
exps["exp-8-b"] = marge({'width': ..., 'depth': ..., 'base_hidden_res', ...}) #smaller

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, help=f"Name of experiment - choose from: {list(exps.keys())}", required=True)
parser.add_argument("--process_limit", type=int, help=f"Max number of processes to run at once", default=4)
args = parser.parse_args()

if __name__ == "__main__":
    t = TPUMaker()
    t.set_project("youdreamof-1543654322305")
    t.set_zone("europe-west4-a")

    exp_name = args.exp_name
    exp = exps[exp_name]

    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(filename=f"logs/{exp_name}.log",
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)

    path = "Metroplex/configs/exps/"
    exp_path = path + exp_name
    os.makedirs(exp_path, exist_ok=True)
    make_configs(exp, exp_path)

    config_list = os.listdir(exp_path)
    config_list = [exp_path.replace("Metroplex/", "") + '/' + config for config in config_list]
    print(f'RUNNING EXPERIMENTS: {config_list}')
    config_chunks = split_list(config_list, args.process_limit)
    processes = []
    names = []

    for chunk_no, config_list in enumerate(config_chunks):
        for idx, config in enumerate(config_list):
            name = f"metroplex-exps-{exp_name}-{chunk_no}-{idx}"
            num_cores = 32
            if not t.tpu_exists(name):
                while True:
                    t.make_tpu(size=num_cores, name=name)
                    if t.tpu_exists(name):
                        break
            cmd = f'python3 run_experiment.py --model {config} --steps_per_checkpoint 1000 --tpu {name} --experiment_name {name}; ' \
                  f'pu delete {name} -y'
            p = subprocess.Popen(cmd, cwd='Metroplex', shell=True)
            processes.append(p)
            names.append(name)

        for p, n in zip(processes, names):
            p.wait()
            out, err = p.communicate()
            logging.info(err)
            logging.info(out)
