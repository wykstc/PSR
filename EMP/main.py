import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from model import MInterface
from data import DInterface
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_callbacks():
    callbacks = []
    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(logging_interval='epoch'))
    return callbacks


def main(args):
    pl.seed_everything(args.seed)
    load_path = None
    data_module = DInterface(**vars(args))
    model = MInterface(**vars(args))


    args.callbacks = load_callbacks()
    trainer = Trainer.from_argparse_args(args)

    #model = MInterface.load_from_checkpoint(checkpoint_path="")
    #trainer.test(model, data_module)
    trainer.fit(model, data_module)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--trainbatch_size', default=20, type=int)
    parser.add_argument('--valbatch_size', default=125, type=int)
    parser.add_argument('--testbatch_size', default=1000, type=int)
    parser.add_argument('--num_workers', default=13, type=int)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)

    parser.add_argument('--lr_scheduler', choices='step', type=str)
    parser.add_argument('--lr_decay_steps', default=10, type=int)
    parser.add_argument('--lr_decay_rate', default=0.1, type=float)
    parser.add_argument('--lr_decay_min_lr', default=3e-12, type=float)

    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--load_v_num', default=None, type=int)

    parser.add_argument('--dataset', default='standard_data', type=str)
    parser.add_argument('--data_dir', default='ref/data', type=str)
    parser.add_argument('--model_name', default='standard_net', type=str)
    parser.add_argument('--loss', default='MSE', type=str)
    parser.add_argument('--weight_decay', default=1e-8, type=float)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--log_dir', default='lightning_logs', type=str)

    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(max_epochs=20)
    args = parser.parse_args()

    main(args)
