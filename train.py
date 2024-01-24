import argparse

import pytorch_lightning as pl

from voca.config import Config
from voca.model import VocaModel
from voca.daatset import VocaDataset


def main():
    config = Config().parse_from_file()
    dataset = VocaDataset(config)
    train_dataloader = dataset.train_dataloader()
    val_dataloader = dataset.val_dataloader()
    
    model = VocaModel(config)
    
    trainer = pl.Trainer(    )
    
    trainer.fit(model, train_dataloader, val_dataloader)
    