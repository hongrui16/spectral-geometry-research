"""Unified three-paradigm trainer with spectral instrumentation.

  python scripts/train.py --objective sft  --model Qwen/Qwen3.5-0.8B ...
  python scripts/train.py --objective opd  --teacher Qwen/Qwen3.5-2B ...
  python scripts/train.py --objective rlvr --rollouts 8 ...

Same base checkpoint, same GSM8K prompt stream (seeded), same optimizer step
count across objectives. Captures G_t / G_b / H_t / W_t on tracked matrices
every --save-every steps (see specgeom.instrument).
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from specgeom.data import PromptSampler, build_prompt, load_gsm8k, reward_fn
from specgeom.instrument import Instrumenter
from specgeom.intervene_engine import InterventionEngine
from specgeom.modeling import decoder_param_groups, load_model
from specgeom.muon import Muon


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--objective", choices=["sft", "opd", "rlvr"], required=True)
    p.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--teacher", default="Qwen/Qwen3.5-2B")
    p.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--muon-lr", type=float, default=2e-4)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--prompts-per-step", type=int, default=8)
    p.add_argument("--rollouts", type=int, default=8, help="K for GRPO")
    p.add_argument("--microbatches", type=int, default=8,
                   help="(unused, kept for compat)")
    p.add_argument("--seqs-per-microbatch", type=int, default=0,
                   help="0 = auto: rlvr 2, sft/opd 1 (memory-bound by "
                        "seqs x seq_len x 248k-vocab fp32 logits)")
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--rollout-capture-mats", type=int, default=4,
                   help="how many tracked matrices get per-rollout grads (RLVR)")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--save-ckpt-every", type=int, default=250)
    p.add_argument("--intervention", default="none",
                   choices=["none", "spectrum_only", "frame_only",
                            "mag_topq", "snr_topq", "adaptive_alpha"])
    p.add_argument("--intervention-q", type=float, default=0.1)
    return p.parse_args()


def batched(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


class Trainer:
    def __init__(self, args):
        self.args = args
        torch.manual_seed(args.seed)
        os.makedirs(args.out, exist_ok=True)
        self.model, self.tok, n_layers = load_model(args.model)
        self.device = next(self.model.parameters()).device
        self.instr = Instrumenter(self.model, args.out, n_layers,
                                  save_every=args.save_every)
        # per-rollout capture subset: spread across layers/types
        names = self.instr.names()
        stride = max(1, len(names) // args.rollout_capture_mats)
        self.rollout_names = names[::stride][:args.rollout_capture_mats]

        matrix, other = decoder_param_groups(self.model)
        if args.optimizer == "adamw":
            self.opt = torch.optim.AdamW(
                [{"params": matrix}, {"params": other}],
                lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
            self.opt_other = None
        else:
            self.opt = Muon(matrix, lr=args.muon_lr, momentum=0.95)
            self.opt_other = torch.optim.AdamW(other, lr=args.lr, betas=(0.9, 0.95))

        self.engine = None
        if args.intervention != "none":
            self.engine = InterventionEngine(self.model, args.intervention,
                                             q=args.intervention_q)

        if args.objective == "opd":
            self.teacher, _, _ = load_model(args.teacher, dtype=torch.bfloat16,
                                            trainable=False)
        data = load_gsm8k("train")
        self.sampler = PromptSampler(data, seed=args.seed)
        self.log_path = os.path.join(args.out, "log.jsonl")
        with open(os.path.join(args.out, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    # ---------------- shared helpers ----------------

    def log(self, rec):
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def tokenize_prompt_completion(self, prompt_ids, completion_ids):
        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + list(completion_ids)
        return input_ids, labels

    def collate(self, rows):
        maxlen = max(len(r[0]) for r in rows)
        pad = self.tok.pad_token_id
        input_ids = torch.full((len(rows), maxlen), pad, dtype=torch.long)
        labels = torch.full((len(rows), maxlen), -100, dtype=torch.long)
        attn = torch.zeros((len(rows), maxlen), dtype=torch.long)
        for i, (ids, lab) in enumerate(rows):
            input_ids[i, :len(ids)] = torch.tensor(ids)
            labels[i, :len(lab)] = torch.tensor(lab)
            attn[i, :len(ids)] = 1
        return (input_ids.to(self.device), labels.to(self.device), attn.to(self.device))

    def lm_logits(self, input_ids, attn):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model(input_ids=input_ids, attention_mask=attn)
        return out.logits.float()

    @torch.no_grad()
    def generate(self, prompts, num_return_sequences=1, greedy=False):
        enc = self.tok(prompts, return_tensors="pt", padding=True,
                       padding_side="left").to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gen = self.model.generate(
                **enc,
                do_sample=not greedy,
                temperature=self.args.temperature if not greedy else None,
                top_p=1.0 if not greedy else None,
                top_k=0 if not greedy else None,
                max_new_tokens=self.args.max_new_tokens,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tok.pad_token_id,
            )
        prompt_len = enc["input_ids"].shape[1]
        comp_ids = gen[:, prompt_len:]
        texts = self.tok.batch_decode(comp_ids, skip_special_tokens=True)
        # strip padding from each completion
        comp_lists = []
        for row in comp_ids:
            ids = row.tolist()
            while ids and ids[-1] == self.tok.pad_token_id:
                ids.pop()
            comp_lists.append(ids)
        # prompt ids per returned sequence (repeat for K)
        prompt_lists = []
        for i in range(gen.shape[0]):
            src = enc["input_ids"][i // num_return_sequences]
            mask = enc["attention_mask"][i // num_return_sequences].bool()
            prompt_lists.append(src[mask].tolist())
        return prompt_lists, comp_lists, texts

    # ---------------- objective losses ----------------

    def sft_rows(self):
        exs = self.sampler.next(self.args.prompts_per_step)
        rows = []
        for ex in exs:
            prompt = build_prompt(self.tok, ex["question"])
            p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
            c_ids = self.tok(ex["solution"] + self.tok.eos_token,
                             add_special_tokens=False)["input_ids"]
            rows.append(self.tokenize_prompt_completion(p_ids, c_ids))
        return rows, {}

    def opd_rows(self):
        exs = self.sampler.next(self.args.prompts_per_step)
        prompts = [build_prompt(self.tok, ex["question"]) for ex in exs]
        p_lists, c_lists, _ = self.generate(prompts)
        rows = [self.tokenize_prompt_completion(p, c)
                for p, c in zip(p_lists, c_lists) if len(c) > 0]
        return rows, {}

    def rlvr_rows(self):
        K = self.args.rollouts
        exs = self.sampler.next(self.args.prompts_per_step)
        prompts = [build_prompt(self.tok, ex["question"]) for ex in exs]
        p_lists, c_lists, texts = self.generate(prompts, num_return_sequences=K)
        all_rewards = []
        # keep only informative groups (nonzero reward variance): degenerate
        # groups have A=0 everywhere and contribute zero gradient; at ~10%
        # reward rate they would otherwise dominate save-step captures
        groups = []  # (rows, advs, texts, complete)
        for gi, ex in enumerate(exs):
            rs = [reward_fn(texts[gi * K + k], ex["gold"]) for k in range(K)]
            all_rewards += rs
            rs_t = torch.tensor(rs, dtype=torch.float32)
            if rs_t.std(unbiased=False) < 1e-6:
                continue
            a = (rs_t - rs_t.mean()) / (rs_t.std(unbiased=False) + 1e-4)
            g_rows, g_advs, g_txt, complete = [], [], [], True
            for k in range(K):
                i = gi * K + k
                if len(c_lists[i]) == 0:
                    complete = False
                    continue
                g_rows.append(self.tokenize_prompt_completion(p_lists[i], c_lists[i]))
                g_advs.append(a[k].item())
                g_txt.append((texts[i], rs[k]))
            groups.append((g_rows, g_advs, g_txt, complete))
        # put a complete group first so the H3 per-rollout capture sees one
        # full K-rollout group with nonzero advantage variance
        groups.sort(key=lambda g: not g[3])
        rows = [r for g in groups for r in g[0]]
        advs_k = [a for g in groups for a in g[1]]
        info = {"reward_mean": float(torch.tensor(all_rewards).mean()),
                "adv": advs_k, "n_seqs": len(rows),
                "n_groups_kept": len(groups),
                "group0_ok": bool(groups) and groups[0][3],
                "group0_texts": groups[0][2] if groups else []}
        return rows, info

    def loss_on(self, rows, info, idxs):
        """Loss for the microbatch given by row indices."""
        sub = [rows[i] for i in idxs]
        input_ids, labels, attn = self.collate(sub)
        logits = self.lm_logits(input_ids, attn)
        logp = F.log_softmax(logits[:, :-1], dim=-1)
        tgt = labels[:, 1:]
        mask = tgt.ne(-100)
        tok_logp = torch.gather(
            logp, 2, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1) * mask

        if self.args.objective in ("sft",):
            return -tok_logp.sum() / mask.sum().clamp_min(1)

        if self.args.objective == "rlvr":
            adv = torch.tensor([info["adv"][i] for i in idxs],
                               device=self.device)
            seq_mean_logp = tok_logp.sum(1) / mask.sum(1).clamp_min(1)
            return -(adv * seq_mean_logp).mean()

        # opd: token-level reverse KL to teacher on student rollouts
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            t_logits = self.teacher(input_ids=input_ids,
                                    attention_mask=attn).logits
        t_logp = F.log_softmax(t_logits.float()[:, :-1], dim=-1)
        s_logp = logp
        kl = (s_logp.exp() * (s_logp - t_logp)).sum(-1) * mask   # (B, T-1)
        return kl.sum() / mask.sum().clamp_min(1)

    def logp_sum_single(self, row):
        """Sum log-prob of one sequence (for per-rollout grad capture)."""
        input_ids, labels, attn = self.collate([row])
        logits = self.lm_logits(input_ids, attn)
        logp = F.log_softmax(logits[:, :-1], dim=-1)
        tgt = labels[:, 1:]
        mask = tgt.ne(-100)
        tok_logp = torch.gather(
            logp, 2, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1) * mask
        return tok_logp.sum()

    # ---------------- main loop ----------------

    def zero_grad(self):
        self.opt.zero_grad(set_to_none=False)
        if self.opt_other:
            self.opt_other.zero_grad(set_to_none=False)

    def step_optimizers(self):
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.args.grad_clip)
        self.opt.step()
        if self.opt_other:
            self.opt_other.step()

    def train(self):
        args = self.args
        make_rows = {"sft": self.sft_rows, "opd": self.opd_rows,
                     "rlvr": self.rlvr_rows}[args.objective]
        for step in range(1, args.steps + 1):
            t0 = time.time()
            self.model.train()
            rows, info = make_rows()
            if len(rows) == 0:
                self.log({"step": step, "skip": "no rows"})
                continue
            save = self.instr.is_save_step(step)
            if save:
                self.instr.begin_step(step)
                # RLVR: per-rollout grad-log-pi for first group (H3 statistic)
                if args.objective == "rlvr" and info.get("group0_ok"):
                    K = args.rollouts
                    n_cap = min(K, len(rows))
                    for k in range(n_cap):
                        self.zero_grad()
                        (-self.logp_sum_single(rows[k])).backward()
                        self.instr.capture_rollout_grad(k, self.rollout_names)
                    self.instr.set_extra(
                        "rollout_adv", [info["adv"][k] for k in range(n_cap)])
                    self.instr.set_extra("rollout_texts",
                                         info.get("group0_texts", []))
                    self.zero_grad()

            mbs = args.seqs_per_microbatch or (2 if args.objective == "rlvr" else 1)
            idx_chunks = [c for c in batched(list(range(len(rows))), mbs)]
            self.zero_grad()
            losses = []
            prev = {n: torch.zeros_like(self.instr.tracked[n], device="cpu")
                    for n in self.instr.snr_names} if save else None
            for bi, chunk in enumerate(idx_chunks):
                loss = self.loss_on(rows, info, chunk) / len(idx_chunks)
                loss.backward()
                losses.append(loss.item() * len(idx_chunks))
                if save:
                    with torch.no_grad():
                        for n in self.instr.snr_names:
                            p = self.instr.tracked[n]
                            cur = p.grad.detach().cpu()
                            gb = (cur - prev[n]) * len(idx_chunks)
                            self.instr._record["mats"][n].setdefault(
                                "G_b", {})[f"b{bi}"] = gb.to(torch.bfloat16)
                            prev[n] = cur

            if save:
                self.instr.capture_grad()
                self.instr.pre_optimizer()
            if self.engine:
                self.engine.observe_grad()
                self.engine.pre_step()
            self.step_optimizers()
            if self.engine:
                self.engine.post_step()
            if save:
                # with an engine, H captured here is the projected update
                self.instr.post_optimizer()
                path = self.instr.flush()

            rec = {"step": step, "loss": sum(losses) / len(losses),
                   "secs": round(time.time() - t0, 2)}
            if "reward_mean" in info:
                rec["reward_mean"] = info["reward_mean"]
                rec["n_groups_kept"] = info.get("n_groups_kept")
            if save and self.engine and self.engine.alpha_log:
                vals = list(self.engine.alpha_log.values())
                rec["alpha_mean"] = sum(vals) / len(vals)
            self.log(rec)
            if step % 10 == 0:
                print(f"[{args.objective}] step {step} "
                      f"loss {rec['loss']:.4f} {rec['secs']}s", flush=True)

            if step % args.save_ckpt_every == 0 or step == args.steps:
                ck = os.path.join(args.out, f"ckpt_{step:06d}")
                self.model.save_pretrained(ck, safe_serialization=True)
                self.tok.save_pretrained(ck)


if __name__ == "__main__":
    args = parse_args()
    Trainer(args).train()
