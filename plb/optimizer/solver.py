from functools import partial
import numpy as np
from core.diffskill.utils import get_component_masks
import plb.utils.utils as tu
import tqdm
import torch
# tu.set_default_tensor_type(torch.DoubleTensor)
from ..engine.function import GradModel
from sklearn.cluster import DBSCAN
FUNCS = {}


class Solver:
    def __init__(self, args, env, ouput_grid=(), **kwargs):
        self.args = args
        self.env = env
        self.env.update_loss_fn(self.args.adam_loss_type)
        self.dbscan = DBSCAN(eps=0.01, min_samples=5)
        if env not in FUNCS:
            FUNCS[env] = GradModel(env, output_grid=ouput_grid, **kwargs)
        self.func = FUNCS[env]
        self.buffer = []

    def solve(self, initial_actions, loss_fn, action_mask=None, lr=0.01, max_iter=200, verbose=True, scheduler=None):
        # loss is a function of the observer ..
        if action_mask is not None:
            initial_actions = initial_actions * action_mask[None]
            action_mask = torch.FloatTensor(action_mask[None]).cuda()
        self.initial_state = self.env.get_state()
        if self.args.component_matching:
            obs_mask, goal_mask = get_component_masks(self.initial_state['state'][0], self.env.target_x, self.dbscan)
            if self.args.debug:
                pcl1, pcl2 = self.initial_state['state'][0][obs_mask], self.env.target_x[goal_mask]
                from chester import logger
                import os
                import matplotlib.pyplot as plt
                fig = plt.figure(figsize=(8,8))
                ax = fig.add_subplot(111, projection='3d')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_zlim(0, 1)
                ax.scatter(pcl1[:,0], pcl1[:,1], pcl1[:,2], marker='o', alpha=0.5)
                ax.scatter(pcl2[:,0], pcl2[:,1], pcl2[:,2], marker='o', alpha=0.5)
                ax.view_init(140,-90)
                plt.savefig(os.path.join(logger.get_dir(), f'components.png'))
                plt.close()

            obs_mask = torch.tensor(obs_mask, dtype=bool, device='cuda')
            goal_mask = torch.tensor(goal_mask, dtype=bool, device='cuda')
            loss_fn = partial(loss_fn, state_mask=obs_mask, goal_mask=goal_mask)
            buffer, info = self.solve_one_plan(initial_actions, loss_fn, action_mask=action_mask,
                                    lr=lr, max_iter=max_iter, verbose=verbose, scheduler=scheduler)
            self.buffer.append(buffer)
            return info, buffer
        if self.args.enumerate_contact:
            ## dbscan the init state; for each component, reset the tool to be on top of the CoM of it, do traj-opt. Return the best trajectory.
            init_pc = self.initial_state['state'][0].reshape(-1, 3)
            self.dbscan.fit(init_pc)
            labels = self.dbscan.labels_
            all_infos = []
            all_buffers = []
            for label in range(labels.max() + 1):
                self.env.set_state(**self.initial_state)
                contact_mask = labels==label
                contact_mask = torch.tensor(contact_mask, dtype=bool, device='cuda')
                cur_loss_fn = partial(loss_fn, state_mask=contact_mask)
                buffer, info = self.solve_one_plan(initial_actions, cur_loss_fn, action_mask=action_mask,
                                    lr=lr, max_iter=max_iter, verbose=verbose, scheduler=scheduler)
                if self.args.debug:
                    from chester import logger
                    import os
                    self.save_plot_buffer(os.path.join(logger.get_dir(), f'solver_loss_{label}.png'), buffer=[buffer])
                    import matplotlib.pyplot as plt
                    print("min max labels:", np.min(self.dbscan.labels_), np.max(self.dbscan.labels_))
                    fig = plt.figure(figsize=(8,8))
                    ax = fig.add_subplot(111, projection='3d')
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.set_zlim(0, 1)
                    ax.scatter(init_pc[:,0], init_pc[:,1], init_pc[:,2], marker='o', alpha=0.5, s=5, c=self.dbscan.labels_)
                    plt.savefig(os.path.join(logger.get_dir(), f'dbscan_{label}.png'))
                    plt.close()
                all_buffers.append(buffer)
                all_infos.append(info)

            improvements = np.array([(all_buffers[i][0]['loss']-all_infos[i]['best_loss']) / all_buffers[i][0]['loss'] for i in range(len(all_infos))])
            best_info = all_infos[np.argmax(improvements)]
            self.buffer.append(all_buffers[all_infos.index(best_info)])
            return best_info, all_buffers[all_infos.index(best_info)]
        else:
            buffer, info = self.solve_one_plan(initial_actions, loss_fn, action_mask=action_mask,
                                    lr=lr, max_iter=max_iter, verbose=verbose, scheduler=scheduler)
            self.buffer.append(buffer)
            return info, buffer


    def solve_one_plan(self, initial_actions, loss_fn, action_mask=None, lr=0.01, max_iter=200, verbose=True, scheduler=None):

        action = torch.nn.Parameter(tu.np2th(np.array(initial_actions)), requires_grad=True)
        optim = torch.optim.Adam([action], lr=lr)
        scheduler if scheduler is None else scheduler(self.optim)
        buffer = []
        best_action, best_loss = initial_actions, np.inf

        iter_id = 0
        ran = tqdm.trange if verbose else range
        it = ran(iter_id, iter_id + max_iter)

        loss, last = np.inf, initial_actions
        H = action.shape[0]
        for iter_id in it:
            optim.zero_grad()

            observations = self.func.reset(self.initial_state['state'])
            cached_obs = []
            for idx, i in enumerate(action):
                if H - idx <= self.args.stop_action_n:
                    observations = self.func.forward(idx, i.detach(), *observations)
                else:
                    observations = self.func.forward(idx, i, *observations)
                cached_obs.append(observations)

            loss = loss_fn(list(range(len(action))), cached_obs, self.args.vel_loss_weight, loss_type=self.args.adam_loss_type)
            assert self.args.energy_weight == 0.
            # energy = torch.sum(action.reshape(-1, 6)[:, 3:] ** 2) # Only penalize the rotational velocity
            # loss += self.args.energy_weight * energy
            loss.backward()
            optim.step()
            if scheduler is not None:
                scheduler.step()
            action.data.copy_(torch.clamp(action.data, -1, 1))
            if action_mask is not None:
                action.data.copy_(action.data * action_mask)

            with torch.no_grad():
                loss = loss.item()
                last = action.data.detach().cpu().numpy()
                if loss < best_loss:
                    best_loss = loss
                    best_action = last

            buffer.append({'action': last, 'loss': loss})
            if verbose:
                it.set_description(f"{iter_id}:  {loss}", refresh=True)

        self.env.set_state(**self.initial_state)
        return buffer, {
            'best_loss': best_loss,
            'best_action': best_action,
            'last_loss': loss,
            'last_action': last
        }

    def eval(self, action, render_fn):
        self.env.simulator.cur = 0
        initial_state = self.initial_state
        self.env.set_state(**initial_state)
        outs = []
        import tqdm
        for i in tqdm.tqdm(action, total=len(action)):
            self.env.step(i)
            outs.append(render_fn())

        self.env.set_state(**initial_state)
        return outs

    def plot_buffer(self, buffer=None):
        import matplotlib.pyplot as plt
        plt.Figure()
        if buffer is None:
            buffer = self.buffer
        for buf in buffer:
            losses = []
            for i in range(len(buf)):
                losses.append(buf[i]['loss'])
            plt.plot(range(len(losses)), losses)
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.show()
    
    def save_plot_buffer(self, path, buffer=None):
        import matplotlib.pyplot as plt
        plt.Figure()
        if buffer is None:
            buffer = self.buffer
        for buf in buffer:
            losses = []
            for i in range(len(buf)):
                losses.append(buf[i]['loss'])
            plt.plot(range(len(losses)), losses)
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.savefig(path)
        plt.close()

    def dump_buffer(self, path='/tmp/buffer.pkl'):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, path='/tmp/buffer.pkl'):
        import pickle
        with open(path, 'rb') as f:
            self.buffer = pickle.load(f)