"""Expose only Torch/Pillow and their Python dependencies; never shared physics packages."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[1]
source=Path('/home/ubuntu/workplace_jclin/membench/runtime/isaac60/lib/python3.12/site-packages')
dest=root/'runtime/torch_deps';dest.mkdir(exist_ok=True)
packages={'torch','torchgen','functorch','triton','nvidia','sympy','mpmath','filelock','fsspec','networkx','jinja2','markupsafe','typing_extensions.py','PIL','pillow.libs'}
links=[]
for p in source.iterdir():
    stem=p.name.split('-')[0].lower()
    if p.name in packages or (p.name.endswith('.dist-info') and (stem in {x.lower().removesuffix('.py') for x in packages} or stem.startswith('nvidia_') or stem=='pillow')):
        target=dest/p.name
        if not target.exists(): target.symlink_to(p,target_is_directory=p.is_dir())
        links.append({'name':p.name,'source':str(p)})
assert not any((dest/n).exists() for n in ['mujoco','mujoco_warp','warp','newton','numpy'])
(dest/'manifest.json').write_text(json.dumps(links,indent=2))
print('Isolated dependency links:',len(links))
