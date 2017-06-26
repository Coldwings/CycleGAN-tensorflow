#!/usr/bin/env python3

from config import Config
from model import GANModel
import os

def main(args=None): #pylint: ing
    config = Config()
    args = config()
    if not os.path.isdir(args.checkpointdir):
        os.mkdir(args.checkpointdir)
    if not os.path.isdir(args.sampledir):
        os.mkdir(args.sampledir)
    model = GANModel(args)
    if args.mode == 'train':
        model.train()
    else:
        model.test(args.mode)


if __name__ == '__main__':
    main()
