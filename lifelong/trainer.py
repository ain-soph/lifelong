#!/usr/bin/env python3

from lifelong.configs import Config, config
import trojanvision.trainer

from typing import TYPE_CHECKING
from trojanvision.datasets import ImageSet    # TODO: python 3.10
from trojanvision.models import ImageModel
import argparse
if TYPE_CHECKING:
    pass


def add_argument(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    return trojanvision.trainer.add_argument(parser=parser)


def create(dataset_name: str = None, dataset: ImageSet = None, model: ImageModel = None,
           config: Config = config, **kwargs):
    return trojanvision.trainer.create(dataset_name=dataset_name, dataset=dataset,
                                       model=model, config=config, **kwargs)
