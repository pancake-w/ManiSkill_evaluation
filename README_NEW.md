## Evaluation Policy in ManiSkill simulation environment
### Project Env Installation

```bash 
conda create -n maniskill_evaluation python==3.10 -y
conda activate maniskill_evaluation
pip install -e .
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 -i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cu121

pip install diffusers==0.11.1 # for diffusion policy evaluation
pip install d3rlpy # for rl policy evaluation
pip install coacd # for 3d_assets
pip install gymnasium==0.29.1
```

You Should also download the 3d_assets and modify the `ASSET_3D_PATH` and `CONTAINER_3D_PATH` in [file](mani_skill/examples/real2sim_3d_assets/__init__.py)
```bash
pip install gdown
gdown "https://drive.google.com/uc?id=1FUOCM5mrI0xxzxvYoBBYOZtoCREDe-9n" # url from bingwen
unzip 'policy_evaluation_3d_assets.zip' -d policy_evaluation_3d_assets
rm 'policy_evaluation_3d_assets.zip'
```

### Env render on local computer
```
python -m mani_skill.examples.demo_random_action -e TabletopPickEnv-v1 --render-mode="human"

python -m mani_skill.examples.demo_random_action -e TabletopPickPlaceEnv-v1 --render-mode="human"

python -m mani_skill.examples.demo_random_action_episode --render_mode human -e TabletopPickPlaceEnv-v1 -r panda_wristcam

```

### Evaluation in Tabletop for Diffusion Policy
```bash
# pick task
object_name="tomato"
pick_ckpt_path="/path/pick.ckpt"
pick_obs_normalize_params_path="/path/pick_norm.pkl"
CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_PREALLOCATE=false python -m mani_skill.evaluation.policy_evaluation \
    --model="diffusion_policy" --ckpt_path="${pick_ckpt_path}" \
    -e "TabletopPickEnv-v1" -s 0 --num-episodes 100 --num-envs 5 --save-video --max_episode_len 100 \
    --object_name="$object_name" -r panda -c pd_ee_pose --obs_normalize_params_path="$pick_obs_normalize_params_path"

# pick and place task
object_name="tomato"
container_name="plate"
pick_place_ckpt_path="/path/pickplace_rel.ckpt"
pick_place_obs_normalize_params_path="/path/pickplace_rel_norm.pkl"
CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false python -m mani_skill.evaluation.policy_evaluation \
    --model="diffusion_policy" --ckpt_path="${pick_place_ckpt_path}" \
    -e "TabletopPickPlaceEnv-v1" -s 0 --num-episodes 100 --num-envs 5 --save-video --max_episode_len 100 \
    --object_name="$object_name" --container_name="$container_name" -r panda -c pd_ee_pose \
    --obs_normalize_params_path="$pick_place_obs_normalize_params_path"
```

### Evaluation in Tabletop for CQL

```bash
# pick task
object_name="tomato"
pick_ckpt_path="/path/rl/cql_model.pt"
pick_obs_normalize_params_path="/path/temp/rl/normalization_params.npz"
CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_PREALLOCATE=false python -m mani_skill.evaluation.policy_evaluation \
    --model="cql" --ckpt_path="${pick_ckpt_path}" \
    -e "TabletopPickEnv-v1" -s 0 --num-episodes 100 --num-envs 1 --save-video --max_episode_len 200 \
    --object_name="$object_name" -r panda -c pd_ee_pose --obs_normalize_params_path="$pick_obs_normalize_params_path"

# pick and place task
object_name="tomato"
container_name="plate"
pick_place_ckpt_path="/path/rl/cql_model.pt"
pick_place_obs_normalize_params_path="/path/rl/normalization_params.npz"
CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false python -m mani_skill.evaluation.policy_evaluation \
    --model="cql" --ckpt_path="${pick_place_ckpt_path}" \
    -e "TabletopPickPlaceEnv-v1" -s 0 --num-episodes 100 --num-envs 1 --save-video --max_episode_len 50 \
    --object_name="$object_name" --container_name="$container_name" -r panda -c pd_ee_pose \
    --obs_normalize_params_path="$pick_place_obs_normalize_params_path"
```
