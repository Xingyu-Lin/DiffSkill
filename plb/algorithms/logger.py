import logging
import os

import time
import os
from torch.utils.tensorboard import SummaryWriter as TorchWriter
from datetime import datetime


class SummaryWriter:
    def __init__(self, path: str):
        if not path.endswith("log"):
            path = os.path.join(path, 'log')
        self.path = path
        self.writer = TorchWriter(log_dir=path)

    def write(self, values):
        step = values['log/step']
        for key, val in values.items():
            if key != 'log/step':
                self.writer.add_scalar(key, val, step)


class Logger:
    def __init__(self, path):
        self.path = path
        self.summary_writer = SummaryWriter(path)
        self.prefix = 'train'
        self.keys = ['step', 'reward', 'loss', 'sdf', 'density', 'contact', 'total_iou', 'last_iou']

        with open(self.filepath(), 'w') as f:
            f.write(','.join(self.keys) + '\n')
        self.steps = 0
        self.episode = 0
        self.not_done = True
        self.start = None
        self.all_reset()

    def filepath(self):
        p = os.path.join(self.path, self.prefix)
        if not os.path.exists(p):
            with open(p, 'w') as f:
                pass
        return p

    def reset(self):
        self.episode += 1
        self.values = {
            i: 0 for i in self.keys
        }
        self.values['step'] = 0
        self.not_done = True

    def all_reset(self):
        self.all_values = {
            i: [] for i in self.keys
        }

    def write(self, values):
        with open(self.filepath(), 'a') as f:
            f.write(','.join(str(values[i]) for i in self.keys) + '\n')

    def step(self, state, action, reward, next_state, done, info):
        if self.start is None:
            self.start = time.time()
        assert self.not_done, "please reset logger."
        self.steps += 1
        self.values['step'] = self.steps

        self.values['reward'] += reward
        self.values['last_iou'] = info['incremental_iou']
        self.values['total_iou'] += info['incremental_iou']
        self.values['sdf'] += info['sdf_loss']
        self.values['density'] += info['density_loss']
        self.values['contact'] += info['contact_loss']
        self.values['loss'] += info['loss']

        if done:
            fps = self.steps / (time.time() - self.start)
            now = datetime.now()
            print(f"{now.strftime('%y-%m-%d %H:%M:%S')} STEP: {self.steps}, reward {self.values['reward']} last_iou {self.values['last_iou']}   fps: {fps}")
            self.write(values=self.values)
            # self.summary_writer.write({'log/' + i: k for i, k in self.values.items()})
            self.not_done = False
            for key in self.values.keys():
                self.all_values[key].append(self.values[key])

