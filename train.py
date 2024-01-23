import argparse

from voca.config import Config
from voca.model import VocaModel
from voca.daatset import VocaDataLoader


def main():
    config = Config().parse_from_file()
    dataloader = VocaDataLoader(config)
    model = VocaModel(config)
    
    
    