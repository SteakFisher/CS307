class BoardConfiguration:
    """
    Represents a board state in the rabbit leap puzzle
    """
    def __init__(self, layout, predecessor=None, level=0):
        self.layout = layout
        self.predecessor = predecessor
        self.level = level

class IterativeDeepeningSearcher:
    """
    Implements Iterative Deepening Depth-First Search for rabbit leap puzzle
    """
    def __init__(self):
        self.total_explorations = 0
    
    def locate_gap(self, board_layout):
        """Find the position of the empty space"""
        return board_layout.find('_')
    
    def generate_moves(self, current_config):
        """Generate all possible moves from current configuration"""
        possible_configs = []
        layout = current_config.layout
        gap_idx = self.locate_gap(layout)
        
        # Check each position for valid moves
        for pos, creature in enumerate(layout):
            # Eastward-moving rabbit rules
            if creature == 'E' and pos < gap_idx:
                # Simple step to adjacent empty space
                if pos + 1 == gap_idx:
                    new_layout = layout[:pos] + '_' + layout[pos+1:gap_idx] + 'E' + layout[gap_idx+1:]
                    possible_configs.append(BoardConfiguration(new_layout, current_config, current_config.level + 1))
                # Jump over westward rabbit
                elif pos + 2 == gap_idx and layout[pos+1] == 'W':
                    new_layout = layout[:pos] + '_' + layout[pos+1:gap_idx] + 'E' + layout[gap_idx+1:]
                    possible_configs.append(BoardConfiguration(new_layout, current_config, current_config.level + 1))
            
            # Westward-moving rabbit rules
            elif creature == 'W' and pos > gap_idx:
                # Simple step to adjacent empty space
                if pos - 1 == gap_idx:
                    new_layout = layout[:gap_idx] + 'W' + layout[gap_idx+1:pos] + '_' + layout[pos+1:]
                    possible_configs.append(BoardConfiguration(new_layout, current_config, current_config.level + 1))
                # Jump over eastward rabbit
                elif pos - 2 == gap_idx and layout[pos-1] == 'E':
                    new_layout = layout[:gap_idx] + 'W' + layout[gap_idx+1:pos] + '_' + layout[pos+1:]
                    possible_configs.append(BoardConfiguration(new_layout, current_config, current_config.level + 1))
        
        return possible_configs
    
    def depth_limited_exploration(self, current_config, target_layout, max_depth):
        """Perform depth-limited search"""
        self.total_explorations += 1
        
        # Check if target is reached
        if current_config.layout == target_layout:
            solution_sequence = []
            temp_config = current_config
            while temp_config is not None:
                solution_sequence.insert(0, temp_config.layout)
                temp_config = temp_config.predecessor
            return solution_sequence
        
        # Stop if depth limit reached
        if max_depth <= 0:
            return None
        
        # Explore all possible next moves
        for next_config in self.generate_moves(current_config):
            exploration_result = self.depth_limited_exploration(next_config, target_layout, max_depth - 1)
            if exploration_result is not None:
                return exploration_result
        
        return None
    
    def iterative_deepening_search(self, initial_layout, target_layout):
        """Main IDDFS algorithm implementation"""
        current_depth_limit = 0
        
        while True:
            print(f"Exploring with depth limit: {current_depth_limit}")
            
            # Reset exploration count for this depth
            starting_config = BoardConfiguration(initial_layout)
            search_result = self.depth_limited_exploration(starting_config, target_layout, current_depth_limit)
            
            if search_result is not None:
                print(f'Total nodes explored: {self.total_explorations}')
                return search_result
            
            current_depth_limit += 1
            
            # Safety check to prevent infinite loops
            if current_depth_limit > 50:
                print(f'Total nodes explored: {self.total_explorations}')
                return None

def run_iterative_deepening_solution():
    """Execute the IDDFS solution for rabbit leap puzzle"""
    # Define puzzle parameters
    starting_arrangement = 'EEE_WWW'
    target_arrangement = 'WWW_EEE'
    
    # Initialize and run search
    search_engine = IterativeDeepeningSearcher()
    solution_path = search_engine.iterative_deepening_search(starting_arrangement, target_arrangement)
    
    # Present results
    if solution_path:
        print("Solution found:")
        for move_number, board_state in enumerate(solution_path, 1):
            print(f"Move {move_number}: {board_state}")
    else:
        print("No solution found.")

if __name__ == "__main__":
    run_iterative_deepening_solution()