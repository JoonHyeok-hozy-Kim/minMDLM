import os
import sys
from datetime import datetime
from multiprocessing import Pool, TimeoutError


script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_path, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import PROJECT_ROOT
from tests.sudoku.SudokuGrid import SudokuGrid


def generate_sudoku(num_grids, transform_cnt=20):
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


def generate_one_rare_worker(_): 
    try:
        S_worker = SudokuGrid()
        
        generated_list = S_worker.create_sudoku_grids(
            num_grids=1,
            shuffle_range=True, 
            transform_cnt=5,    
            shuffle_coord=True,
        )
        
        if generated_list:
            return generated_list[0] 
            
    except Exception as e:
        print(f"Worker Error: {e}")
    return None


def generate_rare_sudoku_in_parallel_with_timeout(num_grids_goal=100, worker_timeout_min=60):
    """
    타임아웃을 적용하여 '희귀한' 스도쿠 셋을 생성합니다.
    
    Args:
        num_grids_goal (int): 최종 목표 스도쿠 개수 (예: 100)
        worker_timeout (int): 작업자 1개가 멈출 수 있는 최대 시간 (초) (예: 60초)
    """
    
    # Set file directory
    dataset_dir = os.path.join(PROJECT_ROOT, "tests", "sudoku", "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%HM")
    file_name = f"sudoku_rare_dataset_size_{num_grids_goal}_{timestamp}.txt"
    file_to_write = os.path.join(dataset_dir, file_name)
    
    # Generate rare sudoku grids in parallel
    num_cores = os.cpu_count()
    print(f"Target: Generate {num_grids_goal} rare sudoku grids using {num_cores} cores.")
    print(f"Timeout: {worker_timeout_min * 60} min")
    
    unique_grids_set = set() # Drop duplicates using set
    start_time = datetime.now().strftime("%Y%m%d_%HM")
    print(f"Multiprocessing started at {start_time}")

    with Pool(processes=num_cores) as pool:
        
        while len(unique_grids_set) < num_grids_goal:
            
            # 1. Create asyncronous workers
            async_results = [pool.apply_async(generate_one_rare_worker, (i,)) for i in range(num_cores)]
            
            for res in async_results:
                try:
                    grid_str = res.get(timeout=worker_timeout_min) 
                    
                    if grid_str and grid_str not in unique_grids_set:
                        unique_grids_set.add(grid_str)
                    
                except TimeoutError:
                    print(f"KILL: After {worker_timeout_min} time out.")
                except Exception as e:
                    print(f"Error unspecified : {e}")
                
                if len(unique_grids_set) >= num_grids_goal:
                    break 

    print(f"Tota {len(unique_grids_set)} rare grid(s) created. ({datetime.now().strftime("%Y%m%d_%HM")})")
    
    print(f"Write on file: {file_to_write}")
    try:
        with open(file_to_write, "w", encoding="utf-8") as f:
            for grid_str in unique_grids_set:
                f.write(grid_str + "\n")
                f.flush()
    except Exception as e:
        print(f"Error during writing on file: {e}")


def read_sudoku_file(file_name):
    dataset_dir = os.path.join(PROJECT_ROOT, "tests", "sudoku", "dataset")
    file_to_read = os.path.join(dataset_dir, file_name)
    
    S = SudokuGrid()
    
    try:
        with open(file_to_read, "r", encoding="utf-8") as f:
            for line in f:
                S.feed_data(line.strip())
    except FileNotFoundError:
        print(f"Error : File not found at: {file_to_read}")
    except Exception as e:
        print(f"Error unspecified : {e}")
        
    return S


if __name__ == '__main__':
    # generate_sudoku(1_000_000)
    generate_rare_sudoku_in_parallel_with_timeout(3, 60)