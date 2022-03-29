import os
import hydra
from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

from model.lightning import LightningDLReg
from model.utils import MyModelCheckpoint
import torch

import random
random.seed(7)

from typing import Any


@hydra.main(config_path="conf", config_name="config")
def main(config: DictConfig) -> Any:
    global conf
    conf = config


    # set via CLI hydra.run.dir
    model_dir = os.getcwd()

    # use only one GPU
    gpu = conf.meta.gpu
    if gpu is not None and isinstance(gpu, int):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        torch.cuda.empty_cache()


    model = LightningDLReg(hparams=conf)
    # configure logger
    logger = TensorBoardLogger(model_dir, name='log')

    # model checkpoint callback with ckpt metric logging
    ckpt_callback = MyModelCheckpoint(save_top_k=2,
                                      save_last=True,
                                      monitor='val_metrics/loss',
                                      dirpath=f'{model_dir}/checkpoints/',
                                      verbose=True
                                      )

    trainer = Trainer(default_root_dir=model_dir,
                      logger=logger,
                      callbacks=[ckpt_callback],
                      # resume_from_checkpoint='.{fill_the_path}/checkpoints/last.ckpt',
                      **conf.training.trainer
                      )

    # run training
    trainer.fit(model)


if __name__ == "__main__":

    main()