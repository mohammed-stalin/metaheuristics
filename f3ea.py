import numpy as np
import tsplib95

# Load TSP data using tsplib95
problem = tsplib95.load('berlin52.tsp')
nodes = list(problem.node_coords.values())
num_cities = len(nodes)

# Generate distance matrix
def compute_distance_matrix(nodes):
    num_cities = len(nodes)
    distance_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            distance_matrix[i, j] = np.linalg.norm(np.array(nodes[i]) - np.array(nodes[j]))
    return distance_matrix

distance_matrix = compute_distance_matrix(nodes)

# F3EA TSP Implementation
class F3EA_TSP:
    def __init__(self, distance_matrix, population_size, num_generations):
        self.distance_matrix = distance_matrix
        self.num_cities = distance_matrix.shape[0]
        self.population_size = population_size
        self.num_generations = num_generations
        self.population = [np.random.permutation(self.num_cities) for _ in range(population_size)]

    def evaluate(self, solution):
        distance = 0
        for i in range(len(solution) - 1):
            distance += self.distance_matrix[solution[i], solution[i + 1]]
        distance += self.distance_matrix[solution[-1], solution[0]]  # Return to start
        return distance

    def find(self):
        return self.population[np.argmin([self.evaluate(ind) for ind in self.population])]

    def fix(self, best_solution):
        return best_solution.copy()

    def finish(self, best_solution):
        # Apply local search to improve the solution
        for _ in range(10):  # Arbitrary number of local search iterations
            new_solution = best_solution.copy()
            i, j = np.random.randint(0, self.num_cities, 2)
            new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
            if self.evaluate(new_solution) < self.evaluate(best_solution):
                best_solution = new_solution
        return best_solution

    def exploit(self, best_solution):
        new_population = []
        for _ in range(self.population_size):
            new_solution = best_solution.copy()
            i, j = np.random.randint(0, self.num_cities, 2)
            new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
            new_population.append(new_solution)
        return new_population

    def analyze(self):
        for generation in range(self.num_generations):
            best_solution = self.find()
            best_solution = self.fix(best_solution)
            best_solution = self.finish(best_solution)
            self.population = self.exploit(best_solution)
            current_best = self.find()
            print(f"Generation {generation}, Best Distance: {self.evaluate(current_best)}")
        return self.find()

# Running the F3EA algorithm for TSP
POPULATION_SIZE = 100
NUM_GENERATIONS = 100

f3ea_tsp = F3EA_TSP(distance_matrix, POPULATION_SIZE, NUM_GENERATIONS)
best_solution = f3ea_tsp.analyze()
print("Best Solution:", best_solution)
print("Best Distance:", f3ea_tsp.evaluate(best_solution))
