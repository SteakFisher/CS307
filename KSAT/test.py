import string
import random
import itertools
import numpy as np

class SatisfiabilityProblemGenerator:
    """
    Generates K-SAT problem instances with configurable parameters
    """
    
    def __init__(self, num_clauses, literals_per_clause, total_variables):
        self.num_clauses = num_clauses
        self.literals_per_clause = literals_per_clause
        self.total_variables = total_variables
        
    def construct_variable_universe(self):
        """
        Create the complete set of positive and negative literals
        """
        positive_literals = list(string.ascii_lowercase)[:self.total_variables]
        negative_literals = [char.upper() for char in positive_literals]
        return positive_literals + negative_literals
    
    def generate_problem_instances(self, instance_count=10):
        """
        Create multiple unique K-SAT problem instances
        """
        literal_universe = self.construct_variable_universe()
        problem_collection = []
        possible_clause_combinations = list(itertools.combinations(literal_universe, self.literals_per_clause))
        
        generation_counter = 0
        while generation_counter < instance_count:
            clause_set = random.sample(possible_clause_combinations, self.num_clauses)
            if clause_set not in problem_collection:
                problem_collection.append(list(clause_set))
                generation_counter += 1
        
        return literal_universe, problem_collection

class TruthAssignmentManager:
    """
    Handles creation and manipulation of truth value assignments
    """
    
    @staticmethod
    def create_random_assignment(variable_list, base_variable_count):
        """
        Generate random boolean assignments for all variables
        """
        positive_assignments = list(np.random.choice([0, 1], base_variable_count))
        negative_assignments = [1 - value for value in positive_assignments]
        
        complete_assignment = dict(zip(variable_list, positive_assignments + negative_assignments))
        return complete_assignment
    
    @staticmethod
    def evaluate_clause_satisfaction(problem_instance, truth_assignment):
        """
        Calculate how many clauses are satisfied by the given assignment
        """
        satisfied_clauses = 0
        
        for clause in problem_instance:
            # Clause is satisfied if any literal in it evaluates to True
            if any(truth_assignment[literal] for literal in clause):
                satisfied_clauses += 1
        
        return satisfied_clauses

# Configuration Input
print("Specify the number of clauses in each instance:")
clause_count = int(input())
print("Specify the number of literals per clause:")
literals_count = int(input())
print("Specify the total number of variables:")
variable_count = int(input())

# Initialize problem generator
problem_factory = SatisfiabilityProblemGenerator(clause_count, literals_count, variable_count)
variable_universe, problem_instances = problem_factory.generate_problem_instances()

# Initialize assignment manager
assignment_handler = TruthAssignmentManager()

class LocalSearchOptimizer:
    """
    Collection of local search algorithms for K-SAT optimization
    """
    
    def __init__(self, assignment_manager):
        self.evaluator = assignment_manager
    
    def gradient_ascent_search(self, problem_instance, initial_assignment, current_satisfaction, step_tracker, iteration_count):
        """
        Hill climbing algorithm implementation with recursive improvement
        """
        optimal_assignment = initial_assignment.copy()
        assignment_values = list(initial_assignment.values())
        assignment_variables = list(initial_assignment.keys())
        
        peak_satisfaction = current_satisfaction
        peak_assignment = initial_assignment.copy()
        modified_assignment = initial_assignment.copy()
        
        for variable_index in range(len(assignment_values)):
            iteration_count += 1
            # Toggle variable truth value
            modified_assignment[assignment_variables[variable_index]] = abs(assignment_values[variable_index] - 1)
            satisfaction_score = self.evaluator.evaluate_clause_satisfaction(problem_instance, modified_assignment)
            
            if peak_satisfaction < satisfaction_score:
                step_tracker = iteration_count
                peak_satisfaction = satisfaction_score
                peak_assignment = modified_assignment.copy()
        
        if peak_satisfaction == current_satisfaction:
            penetrance_metric = f"{step_tracker}/{iteration_count - len(assignment_values)}"
            return optimal_assignment, peak_satisfaction, penetrance_metric
        else:
            current_satisfaction = peak_satisfaction
            optimal_assignment = peak_assignment.copy()
            return self.gradient_ascent_search(problem_instance, optimal_assignment, current_satisfaction, step_tracker, iteration_count)
    
    def beam_width_search(self, problem_instance, starting_assignment, search_width, iteration_limit=1000):
        """
        Beam search implementation with configurable width
        """
        active_beam = [(starting_assignment.copy(), self.evaluator.evaluate_clause_satisfaction(problem_instance, starting_assignment))]
        exploration_steps = 0
        
        for current_iteration in range(1, iteration_limit + 1):
            successor_candidates = []
            
            for assignment_state, satisfaction_level in active_beam:
                for variable_key in assignment_state:
                    # Create neighboring state by flipping variable
                    neighboring_state = assignment_state.copy()
                    neighboring_state[variable_key] = abs(neighboring_state[variable_key] - 1)
                    neighbor_satisfaction = self.evaluator.evaluate_clause_satisfaction(problem_instance, neighboring_state)
                    successor_candidates.append((neighboring_state, neighbor_satisfaction))
                    exploration_steps += 1
                    
                    # Immediate termination if perfect solution found
                    if neighbor_satisfaction == len(problem_instance):
                        return neighboring_state, f"{exploration_steps}/{iteration_limit}"
            
            # Retain only the best candidates up to beam width
            successor_candidates.sort(key=lambda candidate: candidate[1], reverse=True)
            active_beam = successor_candidates[:search_width]
            
            # Check for perfect solution in current beam
            for assignment_state, satisfaction_level in active_beam:
                if satisfaction_level == len(problem_instance):
                    return assignment_state, f"{exploration_steps}/{iteration_limit}"
        
        # Return best available solution if no perfect solution found
        best_available_assignment, best_available_score = active_beam[0]
        return best_available_assignment, f"{exploration_steps}/{iteration_limit}"
    
    def adaptive_neighborhood_search(self, problem_instance, starting_assignment, initial_beam_width, iteration_limit=1000):
        """
        Variable Neighborhood Descent with dynamic beam width adjustment
        """
        active_assignment = starting_assignment.copy()
        active_satisfaction = self.evaluator.evaluate_clause_satisfaction(problem_instance, active_assignment)
        exploration_steps = 0
        dynamic_beam_width = initial_beam_width
        
        for current_iteration in range(1, iteration_limit + 1):
            successor_candidates = []
            
            for variable_key in active_assignment:
                neighboring_assignment = active_assignment.copy()
                neighboring_assignment[variable_key] = abs(neighboring_assignment[variable_key] - 1)
                neighbor_satisfaction = self.evaluator.evaluate_clause_satisfaction(problem_instance, neighboring_assignment)
                successor_candidates.append((neighboring_assignment, neighbor_satisfaction))
                exploration_steps += 1
                
                if neighbor_satisfaction == len(problem_instance):
                    return neighboring_assignment, f"{exploration_steps}/{iteration_limit}", dynamic_beam_width
            
            # Select top candidates based on current beam width
            successor_candidates.sort(key=lambda candidate: candidate[1], reverse=True)
            top_candidates = successor_candidates[:dynamic_beam_width]
            
            # Identify the highest scoring candidate
            premier_candidate, premier_score = top_candidates[0]
            
            if premier_score > active_satisfaction:
                active_assignment = premier_candidate.copy()
                active_satisfaction = premier_score
                dynamic_beam_width = initial_beam_width  # Reset beam width upon improvement
                
                if premier_score == len(problem_instance):
                    return active_assignment, f"{exploration_steps}/{iteration_limit}", dynamic_beam_width
            else:
                dynamic_beam_width += 1  # Expand search space if no improvement
        
        # Return current best if no perfect solution found
        return active_assignment, f"{exploration_steps}/{iteration_limit}", dynamic_beam_width

class ExperimentalResultsCollector:
    """
    Manages collection and presentation of algorithm performance results
    """
    
    def __init__(self):
        self.gradient_results = []
        self.beam_search_narrow_results = []
        self.beam_search_wide_results = []
        self.adaptive_neighborhood_results = []
        self.gradient_assignments = []
        self.beam_narrow_assignments = []
        self.beam_wide_assignments = []
        self.adaptive_assignments = []
        self.gradient_penetrance = []
        self.beam_narrow_penetrance = []
        self.beam_wide_penetrance = []
        self.adaptive_penetrance = []
    
    def record_results(self, gradient_data, beam_narrow_data, beam_wide_data, adaptive_data):
        """
        Store results from all algorithm runs
        """
        # Gradient ascent results
        self.gradient_assignments.append(gradient_data[0])
        self.gradient_results.append(gradient_data[1])
        self.gradient_penetrance.append(gradient_data[2])
        
        # Beam search narrow results
        self.beam_narrow_assignments.append(beam_narrow_data[0])
        self.beam_narrow_penetrance.append(beam_narrow_data[1])
        
        # Beam search wide results
        self.beam_wide_assignments.append(beam_wide_data[0])
        self.beam_wide_penetrance.append(beam_wide_data[1])
        
        # Adaptive neighborhood results
        self.adaptive_assignments.append(adaptive_data[0])
        self.adaptive_penetrance.append(adaptive_data[1])
    
    def display_comprehensive_results(self, problem_instances):
        """
        Present detailed results for all algorithm executions
        """
        print("\nComprehensive Algorithm Performance Analysis:")
        print("=" * 80)
        
        for instance_index, instance_data in enumerate(problem_instances, 1):
            print(f'Problem Instance {instance_index}: {instance_data}')
            print(f'Gradient Ascent: {self.gradient_assignments[instance_index-1]}, Efficiency: {self.gradient_penetrance[instance_index-1]}')
            print(f'Beam Search (Width=3): {self.beam_narrow_assignments[instance_index-1]}, Efficiency: {self.beam_narrow_penetrance[instance_index-1]}')
            print(f'Beam Search (Width=4): {self.beam_wide_assignments[instance_index-1]}, Efficiency: {self.beam_wide_penetrance[instance_index-1]}')
            print(f'Adaptive Neighborhood: {self.adaptive_assignments[instance_index-1]}, Efficiency: {self.adaptive_penetrance[instance_index-1]}')
            print('-' * 40)
    
    def generate_performance_summary(self):
        """
        Create a tabular summary of all algorithm performance metrics
        """
        print("\nAlgorithm Performance Summary Table:")
        print("-" * 85)
        print("| Instance | Gradient Ascent | Beam Search (3) | Beam Search (4) | Adaptive VND |")
        print("-" * 85)
        
        for index in range(len(self.gradient_penetrance)):
            print(f"| {index+1:8} | {self.gradient_penetrance[index]:15} | {self.beam_narrow_penetrance[index]:15} | {self.beam_wide_penetrance[index]:15} | {self.adaptive_penetrance[index]:12} |")
        
        print("-" * 85)

def execute_comprehensive_experiment():
    """
    Main experimental execution function
    """
    # Initialize optimization algorithms and results collector
    search_optimizer = LocalSearchOptimizer(assignment_handler)
    results_collector = ExperimentalResultsCollector()
    
    problem_counter = 0
    
    for current_problem in problem_instances:
        problem_counter += 1
        
        # Generate random initial assignment
        initial_truth_assignment = assignment_handler.create_random_assignment(variable_universe, variable_count)
        baseline_satisfaction = assignment_handler.evaluate_clause_satisfaction(current_problem, initial_truth_assignment)
        
        # Execute Gradient Ascent (Hill Climbing)
        optimal_gradient_assignment, gradient_score, gradient_efficiency = search_optimizer.gradient_ascent_search(
            current_problem, initial_truth_assignment, baseline_satisfaction, 1, 1
        )
        
        # Execute Beam Search with width 3
        beam_narrow_assignment, beam_narrow_efficiency = search_optimizer.beam_width_search(
            current_problem, initial_truth_assignment, search_width=3
        )
        
        # Execute Beam Search with width 4
        beam_wide_assignment, beam_wide_efficiency = search_optimizer.beam_width_search(
            current_problem, initial_truth_assignment, search_width=4
        )
        
        # Execute Adaptive Neighborhood Search
        adaptive_assignment, adaptive_efficiency, final_beam_width = search_optimizer.adaptive_neighborhood_search(
            current_problem, initial_truth_assignment, initial_beam_width=3
        )
        
        # Collect results
        results_collector.record_results(
            (optimal_gradient_assignment, gradient_score, gradient_efficiency),
            (beam_narrow_assignment, beam_narrow_efficiency),
            (beam_wide_assignment, beam_wide_efficiency),
            (adaptive_assignment, adaptive_efficiency)
        )
        
        # Display individual problem results
        print(f'Problem Instance {problem_counter}: {current_problem}')
        print(f'Gradient Ascent: {optimal_gradient_assignment}, Efficiency: {gradient_efficiency}')
        print(f'Beam Search (Width=3): {beam_narrow_assignment}, Efficiency: {beam_narrow_efficiency}')
        print(f'Beam Search (Width=4): {beam_wide_assignment}, Efficiency: {beam_wide_efficiency}')
        print(f'Adaptive Neighborhood: {adaptive_assignment}, Efficiency: {adaptive_efficiency}')
        print()
    
    # Generate comprehensive performance summary
    results_collector.generate_performance_summary()

# Execute the complete experimental workflow
if __name__ == "__main__":
    execute_comprehensive_experiment()
