```bash 
pip install -e .
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 -i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cu121

pip install diffusers==0.11.1
pip install d3rlpy
```

You Should also modify the `ASSET_3D_PATH` and `CONTAINER_3D_PATH` in `ManiSkill/mani_skill/examples/real2sim_3d_assets/__init__.py`