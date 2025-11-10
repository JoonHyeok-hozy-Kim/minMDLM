'''
Requirments
1. mdlm.py
 - self.vocab_size
 - self.pad_token_id   # Not needed. Grid size is fixed.
 - self.mask_token_id
 - self.cls_token_id
 - self.sep_token_id
2. test_run.py
 - self.forward(self, dataset)
 - Dataset.map(tokenizer, ...) compatibility
 - self.batch_decode()
'''



class SudokuTokenizer:
    def __init__(self):
        self.vocab_size = 10 + 3  # Digits 0-9 + CLS, SEP, MASK tokens
        self.mask_token_id = 11  # [MASK] token ID
        self.cls_token_id = 12   # [CLS] token ID
        self.sep_token_id = 13   # [SEP] token ID

    def tokenize(self, sudoku_grid):
        pass
    
    