#!/bin/bash
#SBATCH --job-name=llama-smoke
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/runs/slurm-%x-%j.out

source /home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research/slurm/common.sh

$PY - <<'EOF'
import torch, sys
sys.path.insert(0, '.')
from specgeom.modeling import load_model, decoder_param_groups
from specgeom.instrument import select_tracked
from specgeom.data import build_prompt

model, tok, n_layers = load_model('meta-llama/Llama-3.2-3B-Instruct')
print('class:', type(model).__name__, '| layers:', n_layers,
      '| params:', round(sum(p.numel() for p in model.parameters())/1e9, 2), 'B')
tracked = select_tracked(model, n_layers)
fams = {}
for n in tracked:
    fams.setdefault(n.split('.')[-3], []).append(n)
print('tracked:', len(tracked), '| families:', {k: len(v) for k, v in fams.items()})
for n, p in list(tracked.items())[:4]:
    print(' ', n, tuple(p.shape))
mat, oth = decoder_param_groups(model)
print('matrix group:', len(mat), 'other:', len(oth))
prompt = build_prompt(tok, 'What is 2+2?')
print('prompt tail:', repr(prompt[-80:]))
ids = tok(prompt, return_tensors='pt').to(model.device)
with torch.autocast('cuda', dtype=torch.bfloat16):
    gen = model.generate(**ids, max_new_tokens=16, do_sample=False,
                         pad_token_id=tok.pad_token_id)
print('gen:', repr(tok.decode(gen[0, ids['input_ids'].shape[1]:])))
print('LLAMA_SMOKE_OK')
EOF
