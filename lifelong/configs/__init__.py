#!/usr/bin/env python3

import trojanvision.configs
from trojanvision.configs import Config

import os

config_path: dict[str, str] = {
    'package': os.path.dirname(__file__),
    'user': None,
    'project': os.path.normpath('./configs/lifelong/'),
}
config = Config(_base=trojanvision.configs.config, **config_path)
