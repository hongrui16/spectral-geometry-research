# Replicate the exact_iso post_step on GPU for one base matrix: how big is the perturbation W_new - W_after
# relative to a dense step of size ~6.5e-5*||W||, with TF32 on and off, and repeated 10x (accumulation)?
import torch, glob, sys
sys.path.insert(0, '/home/rhong5/research_pro/hand_modeling_pro/spectral_geometry_research')
from specgeom_v3.modeling import load_model
from safetensors import safe_open
base_dir = glob.glob('/scratch/rhong5/dataset/hf_home/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*')[0]
with safe_open(glob.glob(base_dir+'/*.safetensors')[0],'pt') as sf:
    W0 = sf.get_tensor('model.language_model.layers.7.self_attn.q_proj.weight').float().cuda()
print('tf32 matmul flag as imported:', torch.backends.cuda.matmul.allow_tf32, 'precision', torch.get_float32_matmul_precision())
for tf32 in (True, False):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    S_anchor = torch.linalg.svdvals(W0)
    Sb = torch.linalg.svdvals(W0.double()).float()
    print(f'tf32={tf32}: anchor vs fp64 svdvals rel diff {((S_anchor-Sb).norm()/Sb.norm()).item():.2e}')
    W = W0.clone(); g = torch.Generator(device='cuda').manual_seed(0)
    for step in range(1, 11):
        H = torch.randn(W.shape, generator=g, device='cuda') ; H = H / H.norm() * 6.5e-5 * W0.norm()   # dense-sized step
        W_after = W + H
        Ua, _, Vah = torch.linalg.svd(W_after, full_matrices=False)
        W_new = Ua @ torch.diag(S_anchor) @ Vah
        corr = W_new - W_after
        if step in (1, 10):
            print(f'  step {step}: ||H||/||W||={ (H.norm()/W0.norm()).item():.2e}  ||W_new-W_after||/||H||={ (corr.norm()/H.norm()).item():.2f}  ||W-W0||/||W0||={ ((W_new-W0).norm()/W0.norm()).item():.2e}  rel_dS={ ((torch.linalg.svdvals(W_new.double()).float()-Sb).norm()/Sb.norm()).item():.2e}')
        W = W_new
