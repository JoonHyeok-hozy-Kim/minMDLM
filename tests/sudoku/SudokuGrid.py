import random
import numpy as np

class SudokuGrid:
    def __init__(self):
        self.grid_size = 9  # Standard Sudoku grid size is 9x9
        self.grids_list = []
    
    def feed_data(self, grid_string_data):
        for i, grid_str in enumerate(grid_string_data):            
            result = self._validate_grid(grid_str=grid_str, display=False, return_list=True)
            
            if isinstance(result, list):
                self.grids_list.append(result)
            else:
                print(f"Warning: Skipped invalid {i+1}-th grid : {grid_str}")
    
    def print_grid(self, grid):
        print("-----------------------")
        for i, val in enumerate(grid):
            if i % self.grid_size == 0:
                print()
            print(f"{val} ", end="")
        print("\n-----------------------")
    
    def _recursive_fill(self, coordinates, row_sets, col_sets, box_sets, curr_grid, index=0, do_debug=False, shuffle_range=False):
        if index >= len(coordinates):
            return True

        row, col = coordinates[index]
        box_index = (row // 3) * 3 + (col // 3)
        
        curr_range = random.sample(range(1, 10), 9) if shuffle_range else range(1,10)

        for num in curr_range:
            if (num not in row_sets[row] and
                num not in col_sets[col] and
                num not in box_sets[box_index]):

                curr_grid[row * self.grid_size + col] = num
                row_sets[row].add(num)
                col_sets[col].add(num)
                box_sets[box_index].add(num)
                
                if do_debug and index % 50 == 0:
                    self.print_grid(curr_grid)

                if self._recursive_fill(coordinates, row_sets, col_sets, box_sets, curr_grid, index + 1, do_debug):
                    return True

                # Backtrack
                curr_grid[row * self.grid_size + col] = 0
                row_sets[row].remove(num)
                col_sets[col].remove(num)
                box_sets[box_index].remove(num)

        return False
    
    def _generate_one_grid(self, coordinates, new_grid, do_debug=False, shuffle_range=False):
        row_sets = [set() for _ in range(self.grid_size)]
        col_sets = [set() for _ in range(self.grid_size)]
        box_sets = [set() for _ in range(self.grid_size)]        
        
        if self._recursive_fill(coordinates, row_sets, col_sets, box_sets, new_grid, 0, do_debug, shuffle_range):
            return True        
        return False

    def _transform_swap_rows(self, grid_matrix):
        """ Row swap within a band """
        new_grid = grid_matrix.copy()
        
        for band_idx in range(3):
            if random.random() < 0.5:
                rows = list(range(band_idx * 3, (band_idx + 1) * 3)) # Row idx eg: [0, 1, 2]
                random.shuffle(rows) # eg: [2, 0, 1]
                new_grid[band_idx*3:(band_idx+1)*3, :] = new_grid[rows, :]

        if random.random() < 0.5:
                bands = list(range(3)) # Band idx eg: [0, 1, 2]
                random.shuffle(bands)  # eg: [1, 2, 0]
                permuted_rows = np.concatenate([range(b*3, (b+1)*3) for b in bands])
                new_grid = new_grid[permuted_rows, :]
            
        return new_grid

    def _transform_swap_cols(self, grid_matrix):
        """ Column swap within a band """
        new_grid = grid_matrix.copy()
        
        for band_idx in range(3):
            if random.random() < 0.5:
                cols = list(range(band_idx * 3, (band_idx + 1) * 3))
                random.shuffle(cols)
                new_grid[:, band_idx*3:(band_idx+1)*3] = new_grid[:, cols]
                
        if random.random() < 0.5:
            bands = list(range(3))
            random.shuffle(bands)
            permuted_cols = np.concatenate([range(b*3, (b+1)*3) for b in bands])
            new_grid = new_grid[:, permuted_cols]
            
        return new_grid
    
    def _transform_rotate_reflect(self, grid_matrix):
        """ Rotate & flip """        
        op = random.randint(0, 7)
        
        if op == 0:
            return grid_matrix
        elif op == 1:
            return np.rot90(grid_matrix, 1)
        elif op == 2:
            return np.rot90(grid_matrix, 2)
        elif op == 3:
            return np.rot90(grid_matrix, 3)
        elif op == 4:
            return np.fliplr(grid_matrix) # Flip Left-Right
        elif op == 5:
            return np.flipud(grid_matrix) # Flip Up-Down
        elif op == 6:
            return grid_matrix.T # Transpose 
        elif op == 7:
            return np.fliplr(grid_matrix.T) # Anti-Transpose 

    def _random_transformation(self, flat_grid, transform_cnt):
        grid_matrix = np.array(flat_grid).reshape((self.grid_size, self.grid_size))
        
        transform_types = [
            self._transform_swap_rows,
            self._transform_swap_cols,
            self._transform_rotate_reflect,
        ]
        
        transform_queue = []
        
        for transform in transform_types:
            for _ in range(random.randint(1, transform_cnt)):
                transform_queue.append(transform)
        
        random.shuffle(transform_queue)
        for transform in transform_queue:
            grid_matrix = transform(grid_matrix)
        
        return grid_matrix.flatten().tolist()
        
    
    def create_sudoku_grids(self, num_grids, do_debug=False, shuffle_coord=False, shuffle_range=True, transform_cnt=0):
        new_grid_set = set(self.grids_list)
        coordinates = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)]        
        cnt_created = 0        
        
        while cnt_created < num_grids:
            if shuffle_coord:
                random.shuffle(coordinates)
            new_grid = [0] * (self.grid_size * self.grid_size)
            
            if self._generate_one_grid(coordinates, new_grid, do_debug, shuffle_range):
                # Transformation
                if transform_cnt > 0:
                    new_grid = self._random_transformation(new_grid, transform_cnt)
                
                new_grid_str = ''.join(map(str, new_grid))
                
                # Duplicacy check
                if new_grid_str in new_grid_set:
                    print("Duplicate!")
                    continue
                
                # Validity check
                validity = self._validate_grid(grid_str=new_grid_str, display=False, return_list=False)
                if isinstance(validity, bool) and not validity:
                    continue
                
                new_grid_set.add(new_grid_str)
                cnt_created += 1
                
        self.grids_list = list(new_grid_set)
        
    
    def _validate_grid(self, grid_str, display=False, return_list=False):
        result = [] if return_list else True
        row_sets = [set() for _ in range(self.grid_size)]
        col_sets = [set() for _ in range(self.grid_size)]
        box_sets = [set() for _ in range(self.grid_size)]
        prev_row = 0
        if display:
            print("-----------------------")
        
        for i, val in enumerate(grid_str):
            row = i // self.grid_size
            col = i % self.grid_size
            box_index = (row // 3) * 3 + (col // 3)
            if display:
                if row != prev_row:
                    prev_row = row
                    print("")
                print(f"{val} ", end="")
            
            if val == '0':
                # raise ValueError(f"Invalid value {val} found in grid ({row}, {col})")
                return False
            if val in row_sets[row]:
                # raise ValueError(f"Duplicate value {val} found in row {row} of grid {grid_str}")
                return False
            row_sets[row].add(val)
            
            if val in col_sets[col]:
                # raise ValueError(f"Duplicate value {val} found in column {col} of grid {grid_str}")
                return False
            col_sets[col].add(val)
            
            if val in box_sets[box_index]:
                # raise ValueError(f"Duplicate value {val} found in box {box_index} of grid {grid_str}")
                return False
            box_sets[box_index].add(val)
            
            if return_list:
                result.append(val)
                
        if display:
            print("\n-----------------------")
        
        return result


if __name__ == "__main__":
    S = SudokuGrid()
    new_grids = S.create_sudoku_grids(num_grids=100, do_debug=False, shuffle_coord=False, shuffle_range=True, transform_cnt=5)
    # for grid in new_grids:
    #     S.print_grid(grid)
    
    # grids_list = []
    # for grid_str in new_grids:
    #     grids_list.append(S.validate_grid(grid_str, display=False))
    # print(grids_list)
    