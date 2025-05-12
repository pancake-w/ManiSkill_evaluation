## ManiSkill simulation environment
### Project Env Installation

```bash 
conda create -n maniskill python==3.10 -y
conda activate maniskill
pip install -e .
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 -i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cu121
pip install coacd # for 3d_assets
```

You Should also download the 3d_assets and modify the `ASSET_3D_PATH` and `CONTAINER_3D_PATH` in [file](mani_skill/examples/real2sim_3d_assets/__init__.py)
```bash
pip install gdown
gdown "https://drive.google.com/uc?id=1FUOCM5mrI0xxzxvYoBBYOZtoCREDe-9n" # url from bingwen
unzip 'policy_evaluation_3d_assets.zip' -d maniskill_3d_assets
rm 'policy_evaluation_3d_assets.zip'
```

### Env render on local computer
```
python -m mani_skill.examples.demo_random_action -e TabletopPickEnv-v1 --render-mode="human" --shader="rt-fast"

python -m mani_skill.examples.demo_random_action -e TabletopPickPlaceEnv-v1 --render-mode="human" --shader="rt-fast"
```