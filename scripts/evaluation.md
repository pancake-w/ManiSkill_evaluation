### Evaluation in Tabletop for world model for DP
```bash
# single task evaluation
# pick environment
object_name="tomato"
pick_ckpt_path="/path/pick.ckpt"
pick_obs_normalize_params_path="/path/pick_norm.pkl"
CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_PREALLOCATE=false python -m mani_skill.evaluation.policy_evaluation \
    --model="diffusion_policy" --ckpt_path="${pick_ckpt_path}" \
    -e "TabletopPickEnv-v1" -s 0 --num-episodes 100 --num-envs 50 --save-video --max_episode_len 200 \
    --object_name="$object_name" -r panda -c pd_ee_pose --obs_normalize_params_path="$pick_obs_normalize_params_path"

# pick and place environment
object_name="tomato"
container_name="plate"
pick_place_ckpt_path="/path/pickplace_rel.ckpt"
pick_place_obs_normalize_params_path="/path/pickplace_rel_norm.pkl"
CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false python -m mani_skill.evaluation.policy_evaluation \
    --model="diffusion_policy" --ckpt_path="${pick_place_ckpt_path}" \
    -e "TabletopPickPlaceEnv-v1" -s 0 --num-episodes 100 --num-envs 10 --save-video --max_episode_len 200 \
    --object_name="$object_name" --container_name="$container_name" -r panda -c pd_ee_pose \
    --obs_normalize_params_path="$pick_place_obs_normalize_params_path"
```

### Evaluation in Tabletop for world model for CQL

```bash
object_name="tomato"
pick_ckpt_path="/path/rl/cql_model.pt"
pick_obs_normalize_params_path="/path/temp/rl/normalization_params.npz"
CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_PREALLOCATE=false python -m mani_skill.evaluation.policy_evaluation \
    --model="cql" --ckpt_path="${pick_ckpt_path}" \
    -e "TabletopPickEnv-v1" -s 0 --num-episodes 100 --num-envs 1 --save-video --max_episode_len 200 \
    --object_name="$object_name" -r panda -c pd_ee_pose --obs_normalize_params_path="$pick_obs_normalize_params_path"

# pick and place environment
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