import torch as th
import numpy as np
import tqdm
import torch
# from ....engine.function import GradModel
# from ....engine.taichi_env import TaichiEnv
# solver return the solver states..

ENV = None
FUNCS = {}

DEFAULT_DTYPE = torch.float32

def set_default_tensor_type(dtypecls):
    global DEFAULT_DTYPE
    th.set_default_tensor_type(dtypecls)
    if dtypecls is th.DoubleTensor:
        DEFAULT_DTYPE = torch.float64
    else:
        DEFAULT_DTYPE = torch.float32

def np2th(nparr):
    dtype = DEFAULT_DTYPE if nparr.dtype in (np.float64, np.float32) else None
    return th.from_numpy(nparr).to(device='cuda:0', dtype=dtype)


def np2th_cpu(nparr):
    dtype = DEFAULT_DTYPE if nparr.dtype in (np.float64, np.float32) else None
    return th.from_numpy(nparr).to(dtype=dtype)


def solve(env, func,
          initial_actions,
          loss_fn,
          lr=0.01,
          max_iter=200,
          verbose=True,
          scheduler=None,
          action_dims=None,
          state=None,
          early_stop=None,
          compute_loss_in_end=False):

    if state is None:
        assert initial_actions is not None
        iter_id = 0
        optim_buffer = []
        action = torch.nn.Parameter(np2th(np.array(initial_actions)), requires_grad=True)
        optim = torch.optim.Adam([action], lr=lr)
        initial_state = env.get_state()
        scheduler = scheduler if scheduler is None else scheduler(optim)
        last_action, last_loss = best_action, best_loss = initial_actions, np.inf
    else:
        iter_id = state['iter_id']
        optim_buffer = state['optim_buffer']
        action = state['action']
        optim = state['optim']
        initial_state = state['initial_state']
        scheduler = state['scheduler']
        best_action, best_loss = state['best_action'], state['best_loss']
        last_action, last_loss = state['last_action'], state['last_loss']

    ran = tqdm.trange if verbose else range
    it = ran(iter_id, iter_id+max_iter)

    if action_dims is not None:
        if isinstance(action_dims, tuple):
            zero_masks = torch.ones(action.shape[-1], dtype=torch.bool, device=action.device)
            for i in action_dims:
                zero_masks[i] = False

    non_decrease_iters = 0

    for iter_id in it:
        optim.zero_grad()

        loss = 0
        outputs = []
        observations = func.reset(initial_state['state'])

        if compute_loss_in_end:
            obs_array = []

        def calc_loss(idx, observations):
            l = loss_fn(idx, *observations)
            if isinstance(l, tuple):
                outputs.append(l[1])
                return l[0]
            else:
                return l

        for idx, i in enumerate(action):
            observations = func.forward(idx, i, *observations)

            if not compute_loss_in_end:
                loss += calc_loss(idx, observations)
            else:
                obs_array.append(observations)

        if compute_loss_in_end:
            for idx, observations in enumerate(obs_array):
                loss += calc_loss(idx, observations)

        loss.backward()
        optim.step()

        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            a = torch.clamp(action, -1, 1)
            if isinstance(action_dims, tuple):
                a[:, zero_masks] = 0
            elif action_dims == 'gripper': # action_dims == gripper ..
                a1 = a[:, 0:3]
                a2 = a[:, 6:9]
                a = torch.zeros_like(a)
                a[:,[1, 2]] = a[:, [7, 8]] = (a1[:, 1:] + a2[:, 1:])/2
                a[:,0] = a1[:, 0]
                a[:,6] = a2[:, 0]
            action.data[:] = a

            last_loss = loss.item()
            if np.isnan(last_loss):
                print("MEET NAN!!")
                break
            if last_loss < -10000:
                print("LARGE loss, may due to the bugs", loss)
                continue
            last_action = action.data.detach().cpu().numpy()
            if last_loss < best_loss:
                best_loss = last_loss
                best_action = last_action
                non_decrease_iters = 0
            else:
                non_decrease_iters += 1

        optim_buffer.append({'action':last_action, 'loss':last_loss})
        if verbose:
            word = f"{iter_id}: {last_loss:.4f}  {best_loss:.3f}"
            if len(outputs) > 0:
                for i in outputs[0]:
                    out = 0
                    for j in outputs:
                        out += j[i]
                    word += f', {i}: {out:.3f}'
            it.set_description(word, refresh=True)

        if early_stop is not None and non_decrease_iters >= early_stop:
            break

    env.set_state(initial_state)

    return {
        'best_loss':best_loss,
        'best_action':best_action,
        'last_loss': last_loss,
        'last_action': last_action,

        'iter_id': iter_id,
        'optim_buffer': optim_buffer,
        'action': action,
        'optim': optim,
        'initial_state': initial_state,
        'scheduler': scheduler
    }