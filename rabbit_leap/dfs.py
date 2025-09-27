class PuzzleNode:
    """
    Represents a single state in the rabbit leap puzzle
    """
    def __init__(self, board_state, parent_node=None, depth_level=0):
        self.board_state = board_state
        self.parent_node = parent_node
        self.depth_level = depth_level

class RabbitLeapSolver:
    def __init__(self):
        self.visited_configurations = set()
        self.nodes_examined = 0
    
    def find_empty_slot(self, board):
        """Locate the empty position on the board"""
        return board.index('_')
    
    def create_move(self, board, from_pos, to_pos):
        """Create a new board configuration after a move"""
        board_list = list(board)
        board_list[from_pos], board_list[to_pos] = board_list[to_pos], board_list[from_pos]
        return ''.join(board_list)
    
    def calculate_valid_moves(self, current_node):
        """Calculate all valid moves from current position"""
        valid_moves = []
        board = current_node.board_state
        empty_pos = self.find_empty_slot(board)
        
        # Examine each position on the board
        for idx, piece in enumerate(board):
            if piece == 'E' and idx < empty_pos:
                # East rabbit can move right (step or jump)
                if idx + 1 == empty_pos:  # Step move
                    new_board = self.create_move(board, idx, empty_pos)
                    valid_moves.append(PuzzleNode(new_board, current_node, current_node.depth_level + 1))
                elif idx + 2 == empty_pos and board[idx + 1] == 'W':  # Jump move
                    new_board = self.create_move(board, idx, empty_pos)
                    valid_moves.append(PuzzleNode(new_board, current_node, current_node.depth_level + 1))
            
            elif piece == 'W' and idx > empty_pos:
                # West rabbit can move left (step or jump)
                if idx - 1 == empty_pos:  # Step move
                    new_board = self.create_move(board, idx, empty_pos)
                    valid_moves.append(PuzzleNode(new_board, current_node, current_node.depth_level + 1))
                elif idx - 2 == empty_pos and board[idx - 1] == 'E':  # Jump move
                    new_board = self.create_move(board, idx, empty_pos)
                    valid_moves.append(PuzzleNode(new_board, current_node, current_node.depth_level + 1))
        
        return valid_moves
    
    def reconstruct_solution_path(self, final_node):
        """Build the solution path from start to goal"""
        path_sequence = []
        current = final_node
        while current is not None:
            path_sequence.append(current.board_state)
            current = current.parent_node
        return path_sequence[::-1]
    
    def depth_first_search(self, starting_config, target_config):
        """Perform depth-first search to solve the puzzle"""
        root_node = PuzzleNode(starting_config)
        search_stack = [root_node]
        
        while search_stack:
            current_node = search_stack.pop()
            self.nodes_examined += 1
            
            # Check if goal is reached
            if current_node.board_state == target_config:
                print(f'Total nodes explored: {self.nodes_examined}')
                return self.reconstruct_solution_path(current_node)
            
            # Process if not visited
            if current_node.board_state not in self.visited_configurations:
                self.visited_configurations.add(current_node.board_state)
                
                # Add all possible moves to stack (LIFO for DFS)
                next_moves = self.calculate_valid_moves(current_node)
                search_stack.extend(next_moves)
        
        print(f'Total nodes explored: {self.nodes_examined}')
        return None

def execute_depth_first_search():
    """Main execution function for DFS solution"""
    # Initialize puzzle configurations
    start_configuration = 'EEE_WWW'
    goal_configuration = 'WWW_EEE'
    
    # Create solver and find solution
    puzzle_solver = RabbitLeapSolver()
    solution_sequence = puzzle_solver.depth_first_search(start_configuration, goal_configuration)
    
    # Output results
    if solution_sequence:
        print("Solution found:")
        for step_num, configuration in enumerate(solution_sequence, 1):
            print(f"Step {step_num}: {configuration}")
    else:
        print("No solution found.")

if __name__ == "__main__":
    execute_depth_first_search()
