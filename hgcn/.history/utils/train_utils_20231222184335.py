# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    train_utils                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: lei324 <lei324>                            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/12/22 18:18:27 by lei324            #+#    #+#              #
#    Updated: 2023/12/22 18:18:27 by lei324           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.loss


def format_metrics(metrics, split):
    """
    构建日志文件
    """
    return "".join(
        ["{}_}:{:.4f}".format(split, metric_name, metric_val)
         for metric_name, metric_val in metrics.item()]
    )


def get_dir_name(models_dir):
    """Gets a directory to save the model.

    If the directory already exists, then append a new integer to the end of
    it. This method is useful so that we don't overwrite existing models
    when launching new jobs.

    Args:
        models_dir: The directory where all the models are.

    Returns:
        The name of a new directory to save the training logs and model weights.
    """
    if not os.path.exists(models_dir):
        save_dir = os.path.join(models_dir, '0')
        os.makedirs(save_dir)
    else:
        existing_dirs = np.array(
            [d for d in os.listdir(models_dir)
             if os.path.isdir(os.path.join(models_dir, d))
             ]
        ).astype(np.int32)

        if len(existing_dirs) > 0:
            dir_id = str(existing_dirs.max()+1)
        else:
            dir_id = '1'
        save_dir = os.path.join(models_dir, dir_id)
        os.makedirs(save_dir)
    return save_dir


def add_flags_from_config(parser, config_dict):
    def OrNoe(default):
        def func(x):
            if x.lower() == 'none':
                return None
            elif default is None:
                return str(x)
            else:
                return type(default)(x)

    for param in config_dict:
        default, description = config_dict[param]
        try:
            if isinstance(default, dict):
                parser = add_flags_from_config(parser, default)
            elif isinstance(default, list):
                if len(default) > 0:
                    parser.add_argument(
                        f"--{param}",
                        action="append",
                        type=type(default[0]),
                        default=default,
                        help=description
                    )
                else:
                    pass
                    parser.add_argument(
                        f"--{param}", action="append", default=default, help=description)
            else:
                pass
                parser.add_argument(
                    f"--{param}", type=OrNone(default), default=default, help=description)
        except argparse.ArgumentError:
            print(
                f"Could not add flag for param {param} because it was already present."
            )
    return parser
