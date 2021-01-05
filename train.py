# -*- coding: utf-8 -*-

import trojanvision.environ
import lifelong.datasets
import lifelong.models
import trojanvision.trainer
from trojanvision.utils import summary
import argparse

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    lifelong.datasets.add_argument(parser)
    lifelong.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    args = parser.parse_args()

    env = trojanvision.environ.create(**args.__dict__)
    dataset = lifelong.datasets.create(**args.__dict__)
    model = lifelong.models.create(dataset=dataset, **args.__dict__)
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **args.__dict__)

    # for data in dataset.loader['train'][]
    if env['verbose']:
        summary(env=env, dataset=dataset, model=model, trainer=trainer)
    model._train(**trainer)
