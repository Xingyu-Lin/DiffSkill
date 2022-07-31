import taichi
import argparse
import random
import numpy as np
import torch
from core.utils.utils import set_random_seed

from plb.envs import make
from plb.algorithms.logger import Logger

from plb.algorithms.discor.run_sac import train as train_sac
from plb.algorithms.ppo.run_ppo import train_ppo
from plb.algorithms.TD3.run_td3 import train_td3
from plb.optimizer.solver import solve_action
from plb.optimizer.solver_nn import solve_nn
from plb.algorithms.cem.cem import solve_cem

RL_ALGOS = ['sac', 'td3', 'ppo']
DIFF_ALGOS = ['action', 'nn']



def get_args(cmd=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default=DIFF_ALGOS + RL_ALGOS)
    parser.add_argument("--env_name", type=str, default="Move-v1")
    parser.add_argument("--path", type=str, default='./tmp')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sdf_loss", type=float, default=10)
    parser.add_argument("--density_loss", type=float, default=10)
    parser.add_argument("--contact_loss", type=float, default=1)
    parser.add_argument("--soft_contact_loss", action='store_true')

    parser.add_argument("--num_steps", type=int, default=None)

    # differentiable physics parameters
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--softness", type=float, default=666.)
    parser.add_argument("--optim", type=str, default='Adam', choices=['Adam', 'Momentum'])
    if cmd:
        args = parser.parse_args()
    else:
        args = parser.parse_args("")

    return args


def main():  # Standard plb launch through commandline
    args = get_args()
    if args.num_steps is None:
        if args.algo in DIFF_ALGOS:
            args.num_steps = 50 * 200
        else:
            args.num_steps = 500000

    logger = Logger(args.path)
    set_random_seed(args.seed)

    env = make(args.env_name, nn=(args.algo == 'nn'), sdf_loss=args.sdf_loss,
               density_loss=args.density_loss, contact_loss=args.contact_loss,
               soft_contact_loss=args.soft_contact_loss)
    env.seed(args.seed)

    if args.algo == 'sac':
        train_sac(env, args.path, logger, args)
    elif args.algo == 'action':
        solve_action(env, args.path, logger, args)
    elif args.algo == 'ppo':
        train_ppo(env, args.path, logger, args)
    elif args.algo == 'td3':
        train_td3(env, args.path, logger, args)
    elif args.algo == 'nn':
        solve_nn(env, args.path, logger, args)
    elif args.algo == 'cem':
        solve_cem(env, args.path, logger, args)


import json
import os
from chester import logger


def run_task(arg_vv, log_dir, exp_name):  # Chester launch
    args = get_args(cmd=False)
    args.__dict__.update(**arg_vv)

    if args.num_steps is None:
        if args.algo in DIFF_ALGOS:
            args.num_steps = 50 * 200
        else:
            args.num_steps = 500000

    set_random_seed(args.seed)

    if args.chamfer_loss > 0.:
        args.density_loss = args.sdf_loss = 0.
    env = make(args.env_name, nn=(args.algo == 'nn'), sdf_loss=args.sdf_loss,
               density_loss=args.density_loss, contact_loss=args.contact_loss,
               soft_contact_loss=args.soft_contact_loss, chamfer_loss=args.chamfer_loss)
    env.seed(args.seed)

    # Configure logger
    logger.configure(dir=log_dir, exp_name=exp_name)
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # Dump parameters
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2, sort_keys=True)

    solve_cem(env, args)


if __name__ == '__main__':
    main()
