import torch, glob, time
from safetensors import safe_open
base_dir = glob.glob('/scratch/rhong5/dataset/hf_home/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*')[0]
with safe_open(glob.glob(base_dir+'/*.safetensors')[0],'pt') as sf:
    Ws = {k.split('layers.')[-1]: sf.get_tensor(k).float().cuda() for k in sf.keys()
          if k.endswith(('layers.7.self_attn.q_proj.weight','layers.7.mlp.down_proj.weight','layers.7.mlp.up_proj.weight','layers.7.linear_attn.out_proj.weight','layers.7.linear_attn.in_proj_qkvz.weight'))}
for n, W in Ws.items():
    S64 = torch.linalg.svdvals(W.double())
    print(n, tuple(W.shape))
    for label, fn in [
        ("fp32 default", lambda: torch.linalg.svd(W, full_matrices=False)),
        ("fp32 gesvd", lambda: torch.linalg.svd(W, full_matrices=False, driver='gesvd')),
        ("fp32 gesvda", lambda: torch.linalg.svd(W, full_matrices=False, driver='gesvda')),
        ("fp64 default", lambda: torch.linalg.svd(W.double(), full_matrices=False)),
        ("fp64 gesvd", lambda: torch.linalg.svd(W.double(), full_matrices=False, driver='gesvd'))]:
        try:
            torch.cuda.synchronize(); t=time.time(); U,S,Vh = fn(); torch.cuda.synchronize(); dt=time.time()-t
            rec = (U@torch.diag(S)@Vh).float(); errS=((S.double()-S64).norm()/S64.norm()).item(); errW=((rec-W).norm()/W.norm()).item()
            # one exact_iso re-pin step with this driver: perturbation vs a dense-sized step
            H = torch.randn_like(W); H = H/H.norm()*6.5e-5*W.norm(); Wa = W+H
            Ua,_,Vah = fn.__call__() if False else torch.linalg.svd(Wa.to(U.dtype), full_matrices=False, **({'driver':label.split()[1]} if 'ge' in label else {}))
            Wn = (Ua@torch.diag(S)@Vah).float(); corr=((Wn-Wa).norm()/H.norm()).item()
            print(f"   {label:13s} {dt*1000:7.0f} ms  |dS|/S={errS:.1e}  recon={errW:.1e}  repin_corr/||H||={corr:.2f}")
        except Exception as e:
            print(f"   {label:13s} ERROR {str(e)[:60]}")
print("SVD_DRIVERS_DONE")
