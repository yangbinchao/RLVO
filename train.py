# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger
import pytorch_lightning as pl

from config import get_opts

from SC_Depth import SC_Depth
from SC_DepthV2 import SC_DepthV2

if __name__ == '__main__':
    hparams = get_opts()

    if hparams.model_version == 'v1':
        system = SC_Depth(hparams)
    elif hparams.model_version == 'v2':
        system = SC_DepthV2(hparams)

    logger = TestTubeLogger(
        save_dir="ckpts",
        name=hparams.exp_name,
        debug=False,
        create_git_tag=False
    )
    ''' 
    模型保存
    '''
    ckpt_dir = 'ckpts/{}/version_{}'.format(hparams.exp_name, logger.experiment.version)
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                          filename='{epoch}-{val_loss:.4f}',
                                          monitor='val_loss',
                                          mode='min',
                                          save_last=True,
                                          save_weights_only=True,
                                          save_top_k=3)
    '''
    预训练模型加载
    '''
    if hparams.ckpt_path is not None:
        print('load pre-trained model from {}'.format(hparams.ckpt_path))
        system = system.load_from_checkpoint(hparams.ckpt_path)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      limit_train_batches=hparams.epoch_size, # 训练数据
                      num_sanity_val_steps=5,  # 健全性检查
                      callbacks=[checkpoint_callback],
                      logger=logger,
                      weights_summary=None,
                      progress_bar_refresh_rate=1,
                      accelerator="gpu", 
                      devices=2,
                      auto_select_gpus=True, # 自主选择可用GPU 
                      # gpus= hparams.num_gpus, # 可以选择指定GPU或者指定GPU数目 [0], hparams.num_gpus, -1
                      distributed_backend='ddp' if hparams.num_gpus > 1 else None,
                      benchmark=True,
                      # amp_backend='apex', amp_level='O1', 
                      # precision=16
                      )

    trainer.fit(system)
