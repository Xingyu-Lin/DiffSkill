def test_render():
    from plb.envs import make
    from diffskill.train import get_args
    from plb.engine.taichi_env import TaichiEnv

    device = 'cuda'

    args = get_args("")

    obs_channel = len(args.img_mode)

    env = make(args.env_name, nn=(args.algo == 'nn'), sdf_loss=args.sdf_loss,
               density_loss=args.density_loss, contact_loss=args.contact_loss,
               soft_contact_loss=args.soft_contact_loss, chamfer_loss=args.chamfer_loss)

    env.seed(args.seed)
    taichi_env: TaichiEnv = env.unwrapped.taichi_env

    import numpy as np
    render_kwargs = {'mode': 'rgb'}

    env.reset(init_v=19, target_v=1)
    while (1):
        env.render('human')
    # import time
    # st = time.time()
    # for i in range(10):
    #     obs = taichi_env.render(**render_kwargs)
    # import matplotlib.pyplot as plt
    # env.step(np.zeros(18))
    # plt.figure()
    # plt.imshow(obs[:, :, :3])
    # plt.axis('off')
    # plt.show()


if __name__ == '__main__':
    test_render()
