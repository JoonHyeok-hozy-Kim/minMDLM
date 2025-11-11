from datasets import load_dataset
from SudokuGrid import SudokuGrid
from manage_sudoku_files import read_sudoku_file

# Hyperparameters for training
BATCH_SIZE = 64
NUM_EPOCHS = 10
STEPS_PER_EPOCH = 1000
LEARNING_RATE = 1e-5
MAX_SEQ_LEN_FOR_BATCH = 9**2 + 2 # 9x9 Sudoku grid + CLS and SEP tokens
WANDB_LOG = True

# Hyperparameters for validation
N_VALIDATION_SAMPLES = 10_000 # Validation set size

# Hyperparemeters for sampling
NUM_SAMPLES = 1
SAMPLING_STEPS = 200
SAVE_SAMPLE_AS_FILE = True

def run_sudoku_training():    
    # S = read_sudoku_file("sudoku_dataset_size_10000_20251111_0029.txt")
    # cnt = 0
    # for grid in S.grids_list:
    #     if cnt > 10:
    #         break
    #     S.print_grid(grid)
    #     cnt += 1
    S = SudokuGrid()
    rare_grid = "347159628269483157581672493174568932692347815835291764753914286928736541416825379"
    print(S._validate_grid(rare_grid))

if __name__ == "__main__":
    run_sudoku_training()
    