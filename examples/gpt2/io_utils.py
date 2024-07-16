import os
import time
import math
from absl import logging
from absl import flags
import numpy as np
# import tensorflow.compat.v1 as tf
import shutil
import yaml

# PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) # 上一层路径

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "")) #这一层路径

FLAGS = flags.FLAGS


def _convert(value):
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (float, int)):
        if math.isnan(value):
            return None
        return value
    elif np.isscalar(value):
        if isinstance(value, (np.float32, np.float64)):
            return float(value)
        if isinstance(value, (np.int32, np.int64)):
            return int(value)
        raise ValueError("{}: {} is not supported.".format(value, type(value)))
    elif isinstance(value, np.ndarray):
        value_list: list = value.tolist()
        for i, value in enumerate(value_list):
            value_list[i] = _convert(value)
        return value_list
    elif isinstance(value, (list, tuple)):
        value_list = list(value)
        for i, value in enumerate(value_list):
            value_list[i] = _convert(value)
        return value_list
    elif isinstance(value, dict):
        new_dict = dict()
        for sub_key, sub_value in value.items():
            new_dict[sub_key] = _convert(sub_value)
        return new_dict
    else:
        try:
            return str(value)
        except:
            return 'Unknown'


def set_global_seed(seed=2019):
    import tensorflow.compat.v1 as tf
    import random
    assert seed > 0
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def configure_gpu():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True


def configure_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    FLAGS.base_dir = os.path.join(
        log_dir,
        '{}-{}-{}-{}'.format(FLAGS.agent_name, FLAGS.env_name.lower(), FLAGS.seed, time.strftime('%Y-%m-%d-%H-%M-%S'))
    )
    FLAGS.log_dir = FLAGS.base_dir
    os.makedirs(FLAGS.base_dir, exist_ok=True)
    logging.get_absl_handler().use_absl_log_file()
    logging.info("Logging into: %s" % FLAGS.base_dir)


def save_code(save_dir):
    project_dir = PROJECT_DIR

    if os.path.exists(save_dir+'/code'):
        shutil.rmtree(save_dir+ '/code')
    shutil.copytree(project_dir, save_dir + '/code',
                    ignore=shutil.ignore_patterns('*tokenizer','log*', 'out*', '__pycache__','result*', 'figures*', 'data*','checkpoints',
                                                  '.git','*.pyc', '.idea', '.DS_Store','models'),dirs_exist_ok=True)


def _convert_config(config):
    config_copy = {}
    for key, value in config.items():
        if isinstance(key, tuple):
            config_copy[key[1]] = _convert(value)
        else:
            config_copy[key] = _convert(value)

    return config_copy


def save_config(config, save_dir):
    with open(os.path.join(save_dir, "config.yaml"), "w") as file:
        yaml.safe_dump(_convert_config(config), file, default_flow_style=False)
    try:
        yaml.safe_load(open(os.path.join(save_dir, "config.yaml"), "r"))
    except Exception as e:
        print(e)
        raise ValueError("config cannot be yaml dump.\n{}".format(config))


if __name__ == "__main__":
    print(PROJECT_DIR)
