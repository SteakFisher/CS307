import collections

class GameState:
    def __init__(self, configuration, previous=None, move_count=0):
        self.configuration = configuration
        self.previous = previous
        self.move_count = move_count
    
    def __hash__(self):
        return hash(self.configuration)
    
    def __eq__(self, other):
        return self.configuration == other.configuration

def generate_next_configurations(current_state):
    """
    Generate all possible next configurations from current state
    """
    possible_moves = []
    board = current_state.configuration
    gap_position = board.find('_')
    
    # Check all positions for valid moves
    for position in range(len(board)):
        character = board[position]
        
        # East-facing rabbit logic
        if character == 'E' and position < gap_position:
            # Single step move
            if position + 1 == gap_position:
                next_config = board[:position] + '_' + board[position+1:gap_position] + 'E' + board[gap_position+1:]
                possible_moves.append(GameState(next_config, current_state, current_state.move_count + 1))
            # Jump move (over West-facing rabbit)
            elif position + 2 == gap_position and board[position+1] == 'W':
                next_config = board[:position] + '_' + board[position+1:gap_position] + 'E' + board[gap_position+1:]
                possible_moves.append(GameState(next_config, current_state, current_state.move_count + 1))
        
        # West-facing rabbit logic
        elif character == 'W' and position > gap_position:
            # Single step move
            if position - 1 == gap_position:
                next_config = board[:gap_position] + 'W' + board[gap_position+1:position] + '_' + board[position+1:]
                possible_moves.append(GameState(next_config, current_state, current_state.move_count + 1))
            # Jump move (over East-facing rabbit)
            elif position - 2 == gap_position and board[position-1] == 'E':
                next_config = board[:gap_position] + 'W' + board[gap_position+1:position] + '_' + board[position+1:]
                possible_moves.append(GameState(next_config, current_state, current_state.move_count + 1))
    
    return possible_moves

def breadth_first_exploration(initial_config, target_config):
    """
    Breadth-first search implementation for rabbit leap puzzle
    """
    initial_state = GameState(initial_config)
    exploration_queue = collections.deque([initial_state])
    explored_states = set()
    exploration_counter = 0
    
    while exploration_queue:
        current_state = exploration_queue.popleft()
        
        # Check if we reached the goal
        if current_state.configuration == target_config:
            solution_path = []
            temp_state = current_state
            while temp_state is not None:
                solution_path.insert(0, temp_state.configuration)
                temp_state = temp_state.previous
            print(f'Total nodes explored: {exploration_counter}')
            return solution_path
        
        # Process unvisited states
        if current_state not in explored_states:
            explored_states.add(current_state)
            exploration_counter += 1
            
            # Add all possible next states to queue
            next_states = generate_next_configurations(current_state)
            exploration_queue.extend(next_states)
    
    print(f'Total nodes explored: {exploration_counter}')
    return None

def main():
    # Configuration setup
    initial_arrangement = 'EEE_WWW'
    final_arrangement = 'WWW_EEE'
    
    # Execute search
    result = breadth_first_exploration(initial_arrangement, final_arrangement)
    
    # Display results
    if result:
        print("Solution found:")
        for configuration in result:
            print(configuration)
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()