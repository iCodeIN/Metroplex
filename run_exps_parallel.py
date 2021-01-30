import subprocess
import os
from config_maker import make_configs, split_list
import argparse
from make_tpu import TPUMaker
import logging

exps = dict()

# WARNING: don't forget to set the best config after each experiment.

# exp 1: conv vs. transformer---------------

# conv prog_mode = 0
exps["exp-1-a"] = {'width': [768, 1024], 'transformer': False, 'prog_mode': 0}
exps["exp-1-b"] = {'width': 768, 'depth': 12, 'transformer': False, 'prog_mode': 0}

# conv prog_mode = 1 (maybe not running these)
# exps["exp-1-c"] = {'width': [768, 1024], 'transformer': False, 'prog_mode': 1, 'start_res': [16, 32]}
# exps["exp-1-d"] = {'width': 768, 'depth': 12, 'transformer': False, 'prog_mode': 1, 'start_res': [16, 32]}

# trans
exps["exp-1-e"] = {'width': [384, 512], 'attn_block_opt': [0, 1], 'enc_dec_mode': [True, False]}
exps["exp-1-f"] = {'width': 384, 'depth': 12, 'attn_block_opt': [0, 1], 'enc_dec_mode': [True, False]}

# good_config = {...}
# add this to the next config

# exp 2: batch size 
exps["exp-2-a"] = {'batch_size_inv_factor': 4, 'iter_factor': 4} 
exps["exp-2-b"] = {'batch_size_inv_factor': 16, 'iter_factor': 16} 

# exp 3: lr & clip_grad & zero_weights & final_fn
exps["exp-3-a"] = {'lr': [0.0001, 0.001]}
exps["exp-3-b"] = {'gradient_clipping': 1.0}
exps["exp-3-c"] = {'zero_weights': False}
exps["exp-3-d"] = {'final_fn': False}

# exp 4: zdim 
exps["exp-4"] = {'zdim_factor': [8, 64]}

# exp 5: reduce width, depth, base_hidden_res, bottleneck ratio 
exps["exp-5-a"] = {'base_hidden_res': [12, 24]}
exps["exp-5-b"] = {'width': [256, 512]}
exps["exp-5-c"] = {'depth': [16, 32]}

# exp 6: DMOL
exps["exp-6-a"] = {'color_non_ar': True, 'shared_sigma': [True, False]}
# modify for whatever the best config ->

exps["exp-6-b"] = {'color_non_ar': True, 'shared_sigma': ..., 'num_mixtures': [1, 4]}


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

    path = "Transformer-VDVAE/configs/exps/"
    exp_path = path + exp_name
    os.makedirs(exp_path, exist_ok=True)
    make_configs(exp, exp_path)

    config_list = os.listdir(exp_path)
    config_list = [exp_path.replace("Transformer-VDVAE/", "") + '/' + config for config in config_list]
    print(f'RUNNING EXPERIMENTS: {config_list}')
    config_chunks = split_list(config_list, args.process_limit)
    processes = []
    names = []

    for chunk_no, config_list in enumerate(config_chunks):
        for idx, config in enumerate(config_list):
            name = f"vdvae-exps-{exp_name}-{chunk_no}-{idx}"
            num_cores = 32
            if not t.tpu_exists(name):
                while True:
                    t.make_tpu(size=num_cores, name=name)
                    if t.tpu_exists(name):
                        break
            cmd = f'python3 run_experiment.py --model {config} --tpu {name} --experiment_name {name}; ' \
                  f'pu delete {name} -y'
            # fix here
            p = subprocess.Popen(cmd, cwd='Transformer-VDVAE', shell=True)
            processes.append(p)
            names.append(name)

        for p, n in zip(processes, names):
            p.wait()
            out, err = p.communicate()
            logging.info(err)
            logging.info(out)
