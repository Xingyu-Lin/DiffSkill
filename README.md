# DiffSkill
07/30 Release three diffskill environments LiftSpread, GatherMove, CutRearrange. Code on algorithms will be released very soon!

### Prerequsite
1. Install conda environments according to `environment.yml`
2. Install [pykeops](https://www.kernel-operations.io/keops/python/installation.html)
3. Run `./prepare.sh`
4. Download and unzip initial and target configurations of environmetns from [[Google Drive link for datasets (2G)]](https://drive.google.com/file/d/11XZw-p2FX-yvoHMnc_yNO5x7iiLxwlwB/view?usp=sharing)

### Environments
To test the environment, run `python scripts/random_env.py --env_name {env_name}`, where `env_name` can be from {LiftSpread, GatherMove, CutRearrange}.

## Cite
If you find this codebase useful in your research, please consider citing:
```
@inproceedings{lin2022diffskill,
title={DiffSkill: Skill Abstraction from Differentiable Physics for Deformable Object Manipulations with Tools},
author={Xingyu Lin and Zhiao Huang and Yunzhu Li and Joshua B. Tenenbaum and David Held and Chuang Gan},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=Kef8cKdHWpP}}
}
```
