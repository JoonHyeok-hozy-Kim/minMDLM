from datasets import load_dataset
from SudokuGrid import SudokuGrid

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
    dataset = load_dataset(
        'text',
        data_files={'test': 'tests/sudoku/dataset/dev_test_set.csv',},
        # header=False,
        # column_names=['solution_string'],
    )
    
    sudoku_grid = SudokuGrid()
    sudoku_grid.feed_data(dataset['test']['text'])
    print(f"Validated {len(grid_list)} Sudoku grids.")


if __name__ == "__main__":
    run_sudoku_training()
    