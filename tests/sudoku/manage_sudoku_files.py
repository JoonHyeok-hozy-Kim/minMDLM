import os
import sys
from datetime import datetime


script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_path, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import PROJECT_ROOT
from tests.sudoku.SudokuGrid import SudokuGrid


def generate_sudoku(num_grids, transform_cnt=10):
    # Setup for saving samples
    dataset_dir = os.path.join(PROJECT_ROOT, "tests", "sudoku", "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    file_name = f"sudoku_dataset_size_{num_grids}_{timestamp}.txt"
    file_to_write = os.path.join(dataset_dir, file_name)
    
    # Create Sudoku
    S = SudokuGrid()
    S.create_sudoku_grids(num_grids=num_grids, shuffle_range=True, transform_cnt=transform_cnt)
                 
    try:
        with open(file_to_write, "a", encoding="utf-8") as f:
            for i, grid_str in enumerate(S.grids_list):
                f.write(grid_str)
                f.write("\n")
                
                if i % 1000 == 0:
                    f.flush()
    except Exception as e:
        print(f"Error saving samples: {e}")


def read_sudoku_file(file_name):
    dataset_dir = os.path.join(PROJECT_ROOT, "tests", "sudoku", "dataset")
    file_to_read = os.path.join(dataset_dir, file_name)
    
    S = SudokuGrid()
    
    try:
        with open(file_to_read, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    S.feed_data(line)
    except FileNotFoundError:
        print(f"Error : File not found at: {file_to_read}")
    except Exception as e:
        print(f"Error unspecified : {e}")
        
    return S


if __name__ == '__main__':
    generate_sudoku(2_000_000)
    # sudoku_grid = read_sudoku_file("sudoku_dataset_size_10000_20251111_0029.txt")
    # print(len(sudoku_grid.grids_list))