import argparse
import os
from collections import OrderedDict

from tqdm import tqdm

from agent import get_agent
from common import PHASE_TESTING, PHASE_TRAINING, get_config
from dataset import get_dataloader
from utils import cycle


# Use multiple GPUs
# CUDA_VISIBLE_DEVICES=0,1 python3 train.py
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 python3 train.py
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5,6,7 python3 train.py

# Visualization:
# In project directory run: tensorboard --logdir train_log
# On server run: python3 -m tensorboard.main --logdir train_log --port 10086 --host 127.0.0.1
# On google cloud server run: tensorboard --logdir train_log --port 10086 --host 127.0.0.1
# Multiple experiments:
# 1. ln -s train_log/log joint_experiments/exp1_log
# 2. ln -s ...
# 3. tensorboard --logdir joint_experiments --port 10086 --host 127.0.0.1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--continue', dest='cont', action='store_true', help="continue training from checkpoint")
    parser.add_argument('--ckpt', type=str, default='latest', required=False, help="desired checkpoint to restore")
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False, help="specify gpu ids")
    parser.add_argument('--vis', action='store_true', default=False, help="visualize output in training")
    args = parser.parse_args()

    # create experiment config
    config = get_config(args)
    print(config)

    # create network and training agent
    tr_agent = get_agent(config)
    print(tr_agent.net)

    # load from checkpoint if provided
    if args.cont:
        tr_agent.load_ckpt(args.ckpt)

    # create dataloader
    train_loader = get_dataloader(PHASE_TRAINING, batch_size=config.batch_size, num_workers=config.num_workers)
    val_loader = get_dataloader(PHASE_TESTING, batch_size=config.batch_size, num_workers=config.num_workers)
    val_loader_step = get_dataloader(PHASE_TESTING, batch_size=config.batch_size, num_workers=config.num_workers)
    val_loader_step = cycle(val_loader_step)
    # val_loader = cycle(val_loader)

    # start training
    clock = tr_agent.clock
    max_epoch_acc = 0
    for e in range(clock.epoch, config.nr_epochs):
        # begin iteration
        pbar = tqdm(train_loader)
        for b, data in enumerate(pbar):
            # train step
            outputs, losses = tr_agent.train_func(data)

            # visualize
            # if args.vis and clock.step % config.visualize_frequency == 0:
            #     tr_agent.visualize_batch(data, "train", outputs)

            pbar.set_description("EPOCH[{}][{}]".format(e, b))
            pbar.set_postfix(OrderedDict({k: v.item() for k, v in losses.items()}))

            # validation step
            if clock.step % config.val_frequency == 0:
                data = next(val_loader_step)
                outputs, losses = tr_agent.val_func(data)

                # visualize
                # if args.vis and clock.step % config.visualize_frequency == 0:
                #     tr_agent.visualize_batch(data, "validation", outputs)

            clock.tick()

        # save the best accuracy
        epoch_acc = tr_agent.evaluate(val_loader)
        if epoch_acc > max_epoch_acc:
            tr_agent.save_ckpt('best_acc')
            max_epoch_acc = epoch_acc

        tr_agent.update_learning_rate()
        clock.tock()

        if clock.epoch % config.save_frequency == 0:
            tr_agent.save_ckpt()
        tr_agent.save_ckpt('latest')


if __name__ == '__main__':
    main()
