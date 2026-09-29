# Changes to upstream Retaining by Doing code (princeton-pli/retaining-by-doing @ 1228541)
1. core/data.py MMLUDataset.__init__: `data_dir` was undefined (NameError on eval/train split);
   added `data_dir = Path('data')`. No effect on data or scoring.
2. core/training/{sft,rl}.py: attn_implementation 'flash_attention_2' -> 'sdpa' (flash_attn is not
   installed in ~/envs_py310_new). Both are exact attention; only speed / fp rounding differ.
