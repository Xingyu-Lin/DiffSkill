# CUDA_VISIBLE_DEVICES=0 python3 solve.py dataset 0 --output_dir sampled5 --n_steps 200 --on_policy --dataset
import numpy as np
import cv2
import torch
import os
import argparse
import glob
from plb.cut.solve_utils import solve_scene
from plb.cut.sample_utils import init_env, execute

def send_email(message, title='SendViaPython',
               fro='arcuied@galois.ucsd.edu', to='z2huang@eng.ucsd.edu'):
    import smtplib
    from email.message import EmailMessage

    msg = EmailMessage()
    msg.set_content(message)

    msg['Subject'] = title
    msg['From'] = fro
    msg['To'] = to

    # Send the message via our own SMTP server.
    try:
        s = smtplib.SMTP('localhost') # need install sendemail
    except Exception as e:
        print("You may need install sendemail!!!")
        raise e

    s.send_message(msg)
    s.quit()

def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("primitive", type=str)
    parser.add_argument("--output_dir", type=str, default='tmp')
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--max_timesteps", type=int, default=50)
    parser.add_argument("--n_steps", type=int, default=200)
    parser.add_argument("--send_email", action='store_true')
    parser.add_argument("--last_emd", action='store_true')
    parser.add_argument("--contact_weight", default=1., type=float)
    parser.add_argument("--place_manipulator", action='store_true')

    parser.add_argument("--on_policy", action='store_true')
    parser.add_argument("--dataset", action='store_true')

    parser.add_argument("--obey", action='store_true')

    parser.add_argument("--glob", default="*.pkl", type=str)

    parser.add_argument("--left_right", default=None, type=str)

    parser.add_argument("--use_sdf_dists", action='store_true')
    parser.add_argument("--not_clever_init", action='store_true')
    args=parser.parse_args()
    return args

def solve_one(env, start, goal, prim_id, args, clever_init):
    max_iter = args.n_steps
    if not clever_init and prim_id == 0:
        max_iter = 20 # only do 20 step descend..

    return solve_scene(env, start, goal, prim_id,
                      max_timesteps=args.max_timesteps,
                      place_mainpulator=args.place_manipulator,
                      last_emd=args.last_emd,
                      contact_weight=args.contact_weight,
                      use_sdf_dists=args.use_sdf_dists,
                      max_iter=max_iter, place_filename=None, lr=args.lr, clever_init=clever_init)


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    env, state = init_env()

    if args.dataset:
        args.path = sorted(glob.glob(os.path.join(args.path, args.glob)))
    else:
        args.path = args.path.split(',')

    if args.on_policy:
        for paths in np.array(args.path).reshape(-1, 3):
            cur = None
            print(paths)
            for idx, path in enumerate(paths):
                start, goal = torch.load(path)[:2]
                prim_id = int(idx > 0)
                if cur is None:
                    cur = start
                task = path.split('/')[-1].split('.')[0]
                name = os.path.join(args.output_dir, f'{task}_{prim_id}.webm')
                out = solve_one(env, cur, goal, prim_id, args)
                execute(env, out['initial_state'], out['best_action'], filename=name)
                torch.save(out, os.path.join(args.output_dir, f'{task}_{prim_id}.pkl'))
                cur = env.get_state()

                env.set_state(goal)
                img = np.uint8((env.render('rgb', img_size=512)[..., :3] * 255).clip(0, 255))
                cv2.imwrite(os.path.join(args.output_dir, f'{task}_goal.png'), img[..., ::-1])
    elif args.obey:
        for paths in np.array(args.path).reshape(-1, 3):
            print(paths)
            for idx, path in enumerate(paths):
                start, goal = torch.load(path)[:2]
                for prim_id in range(2):
                    if idx == 0 and prim_id == 0:
                        continue
                    task = path.split('/')[-1].split('.')[0]
                    name = os.path.join(args.output_dir, f'{task}_{prim_id}.webm')
                    out = solve_one(env, start, goal, prim_id, args, clever_init=False)
                    execute(env, out['initial_state'], out['best_action'], filename=name)
                    torch.save(out, os.path.join(args.output_dir, f'{task}_{prim_id}.pkl'))
                    env.set_state(goal)
                    img = np.uint8((env.render('rgb', img_size=512)[..., :3] * 255).clip(0, 255))
                    cv2.imwrite(os.path.join(args.output_dir, f'{task}_goal.png'), img[..., ::-1])
    else:
        for path in args.path:
            start, goal = torch.load(path)[:2]

            for prim in args.primitive.split(','):
                prim_id = int(prim)

                task = path.split('/')[-1].split('.')[0]
                name = os.path.join(args.output_dir, f'{task}_{prim}.webm')
                out = solve_one(env, start, goal, prim_id, args, clever_init=not args.not_clever_init)
                execute(env, out['initial_state'], out['best_action'], filename=name)
                torch.save(out, os.path.join(args.output_dir, f'{task}_{prim}.pkl'))

                env.set_state(goal)
                img = np.uint8((env.render('rgb', img_size=512)[..., :3] * 255).clip(0, 255))
                cv2.imwrite(os.path.join(args.output_dir, f'{task}_goal.png'), img[..., ::-1])

    if args.send_email:
        send_email(f"Finished {args.path}", str(args))


if __name__ == '__main__':
    main()
