from plb.envs import make
import os
from diffskill.env_spec import set_render_mode
from diffskill.generate_reset_motion import generate_reset_motion

import argparse

def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--id', default=0, type=int)
    parser.add_argument('--n', default=0, type=int)
    parser.add_argument('--total', default=10, type=int)
    parser.add_argument('--eval_num', default=20, type=int)
    parser.add_argument('--data_path', default='buffer.xz', type=str)
    parser.add_argument('--save_interval', default=100, type=int)
    args=parser.parse_args()
    return args

args = get_args()
os.makedirs(args.data_path, exist_ok=True)

env = make('CutRearrange-v1', nn=False)
set_render_mode(env, 'CutRearrange-v1')

"""
import cv2
img = env.render('rgb', img_size=512)
cv2.imshow('x', img)
cv2.waitKey(0)
"""
#%%
from diffskill.env_spec import get_tool_spec
tool_spec = get_tool_spec(env, 'CutRearrange-v1')

#%%
from diffskill.utils import visualize_dataset

#%%
import tqdm
import glob

trajs = []

for i in range(4):
    out = sorted(glob.glob(f'sols/sol{i}/traj*_*.pkl'))
    for idx, j in enumerate(tqdm.tqdm(out, total=len(out))):
        sol_path = j
        goal_path = ('_'.join(j.split('_')[:-1])+'.pkl').replace('sols/sol', 'data/')
        trajs.append([sol_path, goal_path, j.split('.')[0][-1]])

for i in range(4):
    out = sorted(glob.glob(f'obey/obey{i}/traj*_*.pkl'))
    for idx, j in enumerate(tqdm.tqdm(out, total=len(out))):
        sol_path = j
        goal_path = ('_'.join(j.split('_')[:-1])+'.pkl').replace('obey/obey', 'data/')
        trajs.append([sol_path, goal_path,  j.split('.')[0][-1]])

for i in range(4):
    out = sorted(glob.glob(f'sol_remain{i}/traj*_*.pkl'))
    for idx, j in enumerate(tqdm.tqdm(out, total=len(out))):
        sol_path = j
        goal_path = ('_'.join(j.split('_')[:-1])+'.pkl').replace('sol_remain', 'data/')
        trajs.append([sol_path, goal_path,  j.split('.')[0][-1]])


#%%

tool_spec = get_tool_spec(env, 'CutRearrange-v1')

#%%
from sample_utils import execute
from diffskill.imitation_buffer import ImitationReplayBuffer

buffer = ImitationReplayBuffer(None)

#%%
print(env.env.generating_cached_state)
#%%
import torch
import time
import numpy as np

batch_size = (len(trajs) - args.eval_num + args.total-1)//args.total # leave the first 10 as evaluation

trajs = trajs[batch_size*args.id + eval_num: batch_size*(args.id+1) + eval_num]
n_buffer = 0
tot_traj = 0

for idx, (sol, goal, tid) in enumerate(tqdm.tqdm(trajs, total=len(trajs))):
    if args.n > 0 and idx >= args.n:
        break
    idx = idx + batch_size*args.id + eval_num

    a = torch.load(sol)
    b = torch.load(goal)

    tid = int(tid)
    action_mask = tool_spec['action_masks'][tid]
    contact_loss_mask = tool_spec['contact_loss_masks'][tid]

    initial_state = a['initial_state']
    goal_state = b[1]
    best_action = a['best_action']

    infos = []
    reset_key = {
        'init_v': idx,
        'target_v': idx,
    }

    reset_key['contact_loss_mask'] = contact_loss_mask

    state = env.reset(**reset_key) #TODO: contact loss mask..
    obs = env.render(mode='rgb')  # rgbd observation

    states, obses, actions, rewards, succs, scores = [], [], [], [], [0.], [0.]  # Append 0 for the first frame for succs and scores
    states.append(state.astype(np.float32))
    obses.append(obs)

    T = 50
    total_r = 0

    total_time = 0
    agent_time = 0
    env_time = 0
    st_time = time.time()
    actions = best_action

    _, _, _, info = env.step(np.zeros(best_action[-1].shape[-1]))
    infos = [info]


    agent_time = time.time() - st_time

    import tqdm
    for i in range(T):
        t1 = time.time()
        next_state, reward, _, info = env.step(best_action[i])
        infos.append(info)
        env_time += time.time() - t1
        states.append(next_state)
        obs = env.render(mode='rgb')
        obses.append(obs)
        total_r += reward
        rewards.append(reward)

        # mass_grids.append(info['mass_grid'])
    target_img = env.target_img

    emds = np.array([info['info_emd'] for info in infos])
    info_normalized_performance = np.array([info['info_normalized_performance'] for info in infos])


    total_time = time.time() - st_time
    ret = {'states': np.array(states).astype(np.float32),
           'obses': np.array(obses).astype(np.float32),
           'actions': np.array(actions).astype(np.float32),
           'target_img': target_img,
           'rewards': np.array(rewards),
           'info_rewards': np.array(rewards),
           'info_emds': emds,
           'info_final_normalized_performance': np.array(info_normalized_performance[-1]),
           'info_normalized_performance': info_normalized_performance,
           'info_total_r': total_r,
           'info_total_time': total_time,
           'info_agent_time': agent_time,
           'info_env_time': env_time,
           'action_mask': action_mask}

    ret.update(**reset_key)
    buffer.add(ret)
    tot_traj += 1
    if tot_traj % args.save_interval == 0:
        generate_reset_motion(buffer, env, reset_gripper2=True)
        buffer.save(os.path.join(os.path.join(args.data_path, f'dataset{n_buffer}.xz')))
        visualize_dataset(os.path.join(args.data_path, f'dataset{n_buffer}.xz'),
                          env.cfg.cached_state_path,
                          os.path.join(args.data_path, f'visualization{n_buffer}.gif'),
                          visualize_reset=True,
                          overlay_target=True)

        n_buffer += 1
        buffer = ImitationReplayBuffer(None)

"""
import cv2
img = env.render('rgb', img_size=512)
cv2.imshow('x', obses[-1])
cv2.waitKey(0)
"""

#%%
import os
generate_reset_motion(buffer, env, reset_gripper2=True)
buffer.save(os.path.join(os.path.join(args.data_path, f'dataset{n_buffer}.xz')))
from diffskill.utils import visualize_dataset
visualize_dataset(os.path.join(args.data_path, f'dataset{n_buffer}.xz'),
                  env.cfg.cached_state_path,
                  os.path.join(args.data_path, f'visualization{n_buffer}.gif'),
                  visualize_reset=True,
                  overlay_target=True)
n_buffer += 1
#%%
