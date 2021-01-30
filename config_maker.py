import itertools
from copy import deepcopy
import math
import json


class ConfigMaker:
    def __init__(self):
        pass

    def update(self, new_attr):
        self.new_attr = new_attr
        for k, v in new_attr.items():
            setattr(self, k, v)

    def make_config(self):
        config = {}
        self.new_attr_str = dict2str(self.new_attr)
        config['model_path'] += self.new_attr_str
        config['n_embd'] = self.n_embd


        for k, v in config.items():
            if 'dropout' in k:
                config[k] = dropout_prob

        return config


def dict2str(dict):
    new_str = ""
    for k, v in dict.items():
        new_str += k + ':' + str(v).lower() + '-'
    if new_str[-1] == '-':
        new_str = new_str[:-1]
    return new_str.replace(",", "-") # get rid of any commas, gsutil really doesn't like them


# create all the possible combinations of arg dicts
def all_possible_args(args):
    # create the dict with the same keys with None as the value
    # also convert a value into a list of the value if it's not a list
    relevant_args = {}
    new_args = {}
    for k, v in args.items():
        relevant_args[k] = None
        new_args[k] = v if isinstance(v, list) else [v]
    args = new_args

    list_of_new_args = []
    all_possible_combs = list(itertools.product(*(args.values())))

    for values in all_possible_combs:
        new_args = deepcopy(relevant_args)
        for idx, k in enumerate(args.keys()):
            new_args[k] = values[idx]
        list_of_new_args += [new_args]
    return list_of_new_args


# create config files according to the exp_args
def make_configs(exp_args, dir_path, mesh_shape=None):
    # take experiment arguments, feed them into ConfigMaker and produce
    # the config files for all the possible combinations of experiment arguments.
    # configs = []
    for new_args in all_possible_args(exp_args):
        # print(new_args)
        config_maker = ConfigMaker()
        config_maker.update(new_args)
        config = config_maker.make_config()
        # print(json.dumps(config, indent=4))
        with open(dir_path + '/' + config_maker.new_attr_str + '.json', "w") as outfile:
            json.dump(config, outfile)

            # configs += [config_maker.make_config()]
    # return configs

def split_list(l, n):
    # splits list/string into n size chunks
    return [l[i:i+n] for i in range(0, len(l), n)]