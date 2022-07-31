import torch
import numpy as np
from geomloss import SamplesLoss
from .solve_func import solve as solve_func
from plb.engine.function import GradModel

GRAD_MODEL = {}


def solve(env, init_action, loss_fn, softness=666., *args, **kwargs):
    if env not in GRAD_MODEL:
        GRAD_MODEL[env] = GradModel(env, softness=softness, return_dist=True)
    func = GRAD_MODEL[env]
    func.softness = softness
    return solve_func(env, func, init_action, loss_fn, *args, **kwargs)

sinkhorn = SamplesLoss(loss='sinkhorn', p=1, blur=0.001)


from pyquaternion import Quaternion
import numpy as np

def angvel(q1, q2):
    q1 = Quaternion(w=q1[0], x=q1[1], y=q1[2], z=q1[3])
    q2 = Quaternion(w=q2[0], x=q2[1], y=q2[2], z=q2[3])
    delta_q = (q2 * q1.conjugate).normalised
    delta_q_len = np.linalg.norm(delta_q.vector)
    delta_q_angle = 2 * np.arctan2(delta_q_len, delta_q.real)
    w = delta_q.vector * delta_q_angle
    return w


from plb.engine.taichi_env import TaichiEnv
from .sample_utils import execute
def move_gripper_to(env: TaichiEnv, goal_state, thr=1e-2, *args, **kwargs):
    def get_state():
        return env.primitives[1].get_state(0)

    state = env.get_state()

    actions = []

    def step(a):
        actions.append(a)
        env.step(a)

    while abs(get_state()[-1] - goal_state[-1]) > thr:
        diff = goal_state[-1] - get_state()[-1]
        step([0] * 9 + [-diff * 10])

    while np.linalg.norm(get_state()[:3] - goal_state[:3]) > thr:
        diff = get_state()[:3] - goal_state[:3]
        step([0] * 3 + list(-diff * 10) + [0] * 3 + [0])

    return execute(env, state, actions, *args, **kwargs)

def move_knife_to(env: TaichiEnv, goal_state, thr=1e-2, *args, **kwargs):
    def get():
        return env.primitives[0].get_state(0)
    state = env.get_state()
    actions = []
    def step(a):
        actions.append(a)
        env.step(a)

    while np.linalg.norm(get()[:3] - goal_state) > thr:
        diff = goal_state[:3] - get()[:3]
        step(list(diff * 10) + [0] * 7)
    return execute(env, state, actions, *args, **kwargs)

def place_manipulator(env, state, tool_id, particle_id, reset_primitive=True, place_manipulator=False, *args, **kwargs):
    # pids, the particles that we need to consider ..
    env.set_state(state)

    x = state['state'][0]
    x = x[particle_id]

    bbox = x.min(axis=0), x.max(axis=0)
    if reset_primitive:
        env.primitives[0].set_state(0, [0.5, 0.3, 0.5, 1, 0, 0 ,0])
        env.primitives[1].set_state(0, [0.5, 0.10, 0.5, 0.707, 0, 0.707, 0, 0.18])
    center = x.mean(axis=0)

    if place_manipulator:
        if tool_id == 0:
            return move_knife_to(env, [center[0], 0.35, center[2]], *args, **kwargs)
        elif tool_id == 1:
            width = bbox[1][0] - bbox[0][0]
            width = max(width *1.05, env.primitives[1].minimal_gap)
            return move_gripper_to(env, np.array([center[0], 0.1, center[2]] + [1, 0, 0, 0] +
                                                 [width]), *args, **kwargs)
        else:
            raise NotImplementedError

def solve_scene(env, start, goal, primitive_id,
                max_timesteps=50,
                contact_weight=1.,
                place_filename=None,
                place_mainpulator=True,
                velocity_penalty=0.01,
                last_emd=False,
                use_sdf_dists=False,
                clever_init=True,
                *args, **kwargs):

    goal_state = torch.tensor(goal['state'][0][:, :3], dtype=torch.float32, device='cuda:0')

    def compute_loss(idx, shape, tool, *args, **kwargs):

        if primitive_id == 0:
            dists = shape[:, 6+primitive_id:6+primitive_id+1]
            min_ = dists.min(axis=0)[0].clamp(0.0, 1e9)
            contact_dists = (min_.sum() - 0.0) * contact_weight * 10
        elif primitive_id == 1 and use_sdf_dists:
             dists = shape[:, 6 + primitive_id:6 + primitive_id + 2]
             #print(dists.min(axis=0)[0])
             min_ = dists.min(axis=0)[0].clamp(0.001, 1e9)
             contact_dists = (min_**2).sum() * contact_weight
            # size = 0.06
        else:
            center = tool[1][:3]
            gap = tool[1][7]
            z1 = center[2] - gap/2 + 0.015 # note that for sampled 5 it's 0.015..
            z2 = center[2] + gap/2 - 0.015
            a = torch.stack([center[0], center[1], z1])
            b = torch.stack([center[0], center[1], z2])
            # cc = ((a[None, :] - shape[:, :3]) ** 2).sum(axis=1)
            # print(cc.shape)
            # print(gap, z1, z2, shape[cc.min(axis=0)[1]][:3])
            d1 = ((a[None, :] - shape[:, :3])**2).sum(axis=1).min()
            d2 = ((b[None, :] - shape[:, :3])**2).sum(axis=1).min()
            contact_dists = (d1.clamp(0.00005, 1e9) + d2.clamp(0.00005, 1e9) - 0.0001) * contact_weight

            if last_emd:
                contact_dists += torch.relu(gap - 0.12) * 0.1


        loss = 0
        extra = {}
        if not last_emd or idx == max_timesteps-1:
            if not last_emd:
                rand_idx = np.random.choice(len(shape), 500, replace=False)
            else:
                rand_idx = np.arange(len(shape))

            shape_dists = sinkhorn(shape[rand_idx, :3],
                                   goal_state[rand_idx, :3])
            if last_emd:
                shape_dists *= max_timesteps
                v = (shape[:, 3:6]**2).mean()
                extra['v'] = float(v)
                loss += velocity_penalty * v
        else:
            shape_dists = 0
            if last_emd:
                extra['v'] = 0
        loss = loss + contact_dists + shape_dists
        #shape_dists = (shape[:, :3] - goal_state[:, :3]).mean(axis=0)
        return loss, {'c': float(contact_dists), 'emd': float(shape_dists), **extra}

    assert not place_mainpulator, "we don't support place arbitrary manipulator"
    place_manipulator(env, start, primitive_id, None, place_mainpulator=place_mainpulator, filename=place_filename)
    #env.set_state(start)
    if primitive_id == 0:
        action_dims = (0, 1)
        init = np.zeros((max_timesteps, 3+7))
        if clever_init:
            init[:20, 1] = -0.3
        softness = 666.
        #kwargs['max_iter'] = 300
    else:
        action_dims = (3, 4, 5, 9)
        init = np.zeros((max_timesteps, 3+7))
        softness = 666.

    outs = solve(env, init,
                 compute_loss, *args, action_dims=action_dims,
                 softness=softness,
                 **kwargs)
    return outs