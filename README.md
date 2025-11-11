# Minimal Implementation of MDLM based on DiT Llama
[@sahooSimpleEffectiveMasked2024]

### To-Dos
- [x] Implement DiT-Llama backbone
- [ ] Implement MDLM model
    - [x] Training
        - [x] $`\alpha_t'`$ weight implementation
    - [ ] Sampling
        - [x] `mdlm.sample` implementation
        - [x] main loop sampling logic implementation
        - [x] CPU test
        - [ ] Compare with the author's code.
- [ ] Test
    - [x] Wikipedia dataset generation test
    - [ ] Sudoku problem sovling test
        - [x] Implement sudoku generator
        - [ ] Implement sudoku tokenizer
        - [ ] Conduct test run



### venv settings
```bash
pip install torch transformers datasets wandb tqdm
```

### Run
```bash
python -m main
```