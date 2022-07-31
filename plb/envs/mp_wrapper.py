import torch.multiprocessing as mp
from torch.multiprocessing import Pipe, Process
import pickle
import cloudpickle


class CloudpickleWrapper(object):
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        self.x = pickle.loads(ob)

    def __call__(self):
        return self.x()


def worker(remote, parent_remote, env_fn):
    parent_remote.close()
    env = env_fn()
    while True:
        cmd, data = remote.recv()
        if cmd == 'reset':
            ob = env.reset(**data)
            remote.send(ob)
        elif cmd == 'step':
            ob, reward, done, info = env.step(data)
            remote.send((ob, reward, done, info))
        elif cmd == 'render':
            remote.send(env.render(**data))
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'getattr':
            remote.send(eval('env.' + data))
        elif cmd == 'getfunc':
            name, kwargs = data
            remote.send(eval('env.' + name)(**kwargs))
        else:
            raise NotImplementedError


class SubprocVecEnv(object):
    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False
        no_of_envs = len(env_fns)
        self.remotes, self.work_remotes = \
            zip(*[Pipe() for _ in range(no_of_envs)])
        self.ps = []

        for wrk, rem, fn in zip(self.work_remotes, self.remotes, env_fns):
            proc = Process(target=worker,
                           args=(wrk, rem, CloudpickleWrapper(fn)))
            self.ps.append(proc)

        for p in self.ps:
            p.daemon = True
            p.start()

        for remote in self.work_remotes:
            remote.close()

    def step_async(self, actions):
        if self.waiting:
            assert False, 'Already stepping'
        self.waiting = True

        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

    def step_wait(self):
        if not self.waiting:
            assert False, 'Not stepping'
        self.waiting = False

        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return obs, rews, dones, infos

    def step(self, actions):
        """ Input list of actions"""
        self.step_async(actions)
        return self.step_wait()

    def reset(self, list_kwargs):
        """ Input list of kwargs"""
        for remote, kwargs in zip(self.remotes, list_kwargs):
            remote.send(('reset', kwargs))

        return [remote.recv() for remote in self.remotes]

    def render(self, list_kwargs):
        for remote, kwargs in zip(self.remotes, list_kwargs):
            remote.send(('render', kwargs))
        return [remote.recv() for remote in self.remotes]

    # def primitive_reset_to(self, list_kwargs):
    #     for remote, kwargs in zip(self.remotes, list_kwargs):
    #         remote.send(('reset', kwargs))

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def getattr(self, name, idx=None):
        """ Relay unknown attribute access to the wrapped_env. For now only use the first env"""
        if idx is not None:
            remote = self.remotes[idx]
            remote.send(('getattr', name))
            return remote.recv()
        else:
            for remote in self.remotes:
                remote.send(('getattr', name))
            return [remote.recv() for remote in self.remotes]

    def getfunc(self, name, idx=None, list_kwargs=None):
        if idx is not None:
            if list_kwargs is None:
                kwargs = {}
            else:
                kwargs = list_kwargs[0]
            remote = self.remotes[idx]
            remote.send(('getfunc', (name, kwargs)))
            return remote.recv()
        else:
            if list_kwargs is None:
                list_kwargs = [{}] * len(self.remotes)
            for remote, kwargs in zip(self.remotes, list_kwargs):
                remote.send(('getfunc', (name, kwargs)))
            return [remote.recv() for remote in self.remotes]


from plb.envs import make


def make_mp_envs(env_name, num_env, seed, start_idx=0, return_dist=False):
    def make_env(rank):
        def fn():
            from core.diffskill.env_spec import set_render_mode
            import os
            s = os.environ.get("CUDA_VISIBLE_DEVICES")
            if s is None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
            else:
                device_ids = s.split(',')
                os.environ["CUDA_VISIBLE_DEVICES"] = str(device_ids[int(rank)])

            env = make(env_name, return_dist=return_dist)
            env.seed(seed + rank)
            set_render_mode(env, env_name, 'mesh')
            return env

        return fn

    return SubprocVecEnv([make_env(i + start_idx) for i in range(num_env)])
