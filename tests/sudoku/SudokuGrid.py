import random

class SudokuGrid:
    def __init__(self):
        self.grid_size = 9  # Standard Sudoku grid size is 9x9
        self.grids_list = []
    
    def feed_data(self, grid_string_data):
        validate_grids = []
        for i, grid_str in enumerate(grid_string_data):
            validate_grids.append(self.validate_grid(grid_str, True))
        
        self.grids_list.extend(validate_grids)
    
    def _recursive_fill(self, coordinates, row_sets, col_sets, box_sets, curr_grid, index=0):
        if index >= len(coordinates):
            return True

        row, col = coordinates[index]
        box_index = (row // 3) * 3 + (col // 3)

        for num in random.sample(range(1, 10), 9):
            if (num not in row_sets[row] and
                num not in col_sets[col] and
                num not in box_sets[box_index]):

                curr_grid[row * self.grid_size + col] = num
                row_sets[row].add(num)
                col_sets[col].add(num)
                box_sets[box_index].add(num)

                if self._recursive_fill(coordinates, row_sets, col_sets, box_sets, curr_grid, index + 1):
                    return True

                # Backtrack
                curr_grid[row * self.grid_size + col] = None
                row_sets[row].remove(num)
                col_sets[col].remove(num)
                box_sets[box_index].remove(num)

        return False
    
    def _generate_one_grid(self, coordinates, new_grid):
        row_sets = [set() for _ in range(self.grid_size)]
        col_sets = [set() for _ in range(self.grid_size)]
        box_sets = [set() for _ in range(self.grid_size)]        
        
        if self._recursive_fill(coordinates, row_sets, col_sets, box_sets, new_grid):
            return True        
        return False
    
    
    def create_sudoku_grids(self, num_grids):
        new_grid_str_list = []
        coordinates = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)]
        
        cnt_created = 0
        
        while cnt_created < num_grids:
            random.shuffle(coordinates)
            new_grid = [None] * (self.grid_size * self.grid_size)
            if self._generate_one_grid(coordinates, new_grid):
                new_grid_str_list.append(''.join(map(str, new_grid)))
                cnt_created += 1
                
        return new_grid_str_list
        
    
    def validate_grid(self, grid_str, display=False):
        grid_list = []
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
                raise ValueError(f"Invalid value {val} found in grid ({row}, {col})")
            if val in row_sets[row]:
                raise ValueError(f"Duplicate value {val} found in row {row} of grid {grid_str}")
            row_sets[row].add(val)
            
            if val in col_sets[col]:
                raise ValueError(f"Duplicate value {val} found in column {col} of grid {grid_str}")
            col_sets[col].add(val)
            
            if val in box_sets[box_index]:
                raise ValueError(f"Duplicate value {val} found in box {box_index} of grid {grid_str}")
            box_sets[box_index].add(val)
            
            grid_list.append(val)
                
        if display:
            print("\n-----------------------")
        
        return grid_list


if __name__ == "__main__":
    S = SudokuGrid()
    new_grids = S.create_sudoku_grids(1)
    # print(new_grid)
    
    grids_list = []
    for grid_str in new_grids:
        grids_list.append(S.validate_grid(grid_str, display=True))
    print(grids_list)