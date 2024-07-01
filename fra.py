import random
import math
import tsplib95
import numpy as np

# Constants (These can be adjusted based on the problem)
HYDRAULIC_DIAMETER = 1.0
MASS_FLUX = 1.0
CHARACTERISTIC_LENGTH = 1.0
NTU = 1.0
CHANNEL_HEIGHT = 1.0
SPECIFIC_AREA = 1.0
SCALING_FACTOR = 1.0
MATRIX_THICKNESS = 1.0
VISCOSITY = 1.0
DENSITY = 1.0
POROSITY = 1.0

def levy_distribution():
    """Generates a number using Levy distribution."""
    return np.random.standard_cauchy()

def gaussian_distribution():
    """Generates a number using Gaussian distribution."""
    return np.random.normal()

def reynolds_number(velocity):
    """Calculates the Reynolds number."""
    return (DENSITY * velocity * CHARACTERISTIC_LENGTH) / VISCOSITY

def distance(city1, city2):
    """Calculates the Euclidean distance between two cities."""
    x1, y1 = city1
    x2, y2 = city2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def generate_initial_population(num_individuals, num_cities):
    """Creates a population of candidate solutions."""
    population = []
    for _ in range(num_individuals):
        permutation = random.sample(range(num_cities), num_cities)  # Ensure no duplicates
        population.append(permutation)
    return population

def calculate_fitness(permutation, city_locations):
    """Evaluates the fitness (total distance) of a given route."""
    total_distance = 0
    for i in range(len(permutation) - 1):
        city1 = city_locations[permutation[i]]
        city2 = city_locations[permutation[i + 1]]
        total_distance += distance(city1, city2)
    city1 = city_locations[permutation[-1]]
    city2 = city_locations[permutation[0]]
    total_distance += distance(city1, city2)
    return total_distance

def update_positions(population, city_locations, alpha, beta, gamma):
    """Updates the positions of the individuals based on flow regime principles."""
    new_population = []
    for individual in population:
        new_position = individual[:]
        velocity = gamma * gaussian_distribution()  # Velocity based on Gaussian distribution
        re_number = reynolds_number(velocity)
        for i in range(len(individual)):
            if re_number < 2000:  # Laminar flow
                levy_step = int(levy_distribution())
                if random.random() < alpha:
                    # Flow regime attraction (move towards a better solution)
                    best_neighbor_index = find_best_neighbor(individual, city_locations)
                    new_position[i], new_position[best_neighbor_index] = new_position[best_neighbor_index], new_position[i]
                else:
                    new_position[i] = (new_position[i] + levy_step) % len(city_locations)
            else:  # Turbulent flow
                random_step = int(gaussian_distribution())
                new_position[i] = (new_position[i] + random_step) % len(city_locations)
            if random.random() < beta:
                # Random disturbance to avoid local minima
                random_index = random.randint(0, len(individual) - 1)
                new_position[i], new_position[random_index] = new_position[random_index], new_position[i]
        # Ensure valid TSP solution by removing duplicates and restoring missing cities
        new_position = restore_valid_permutation(new_position, len(city_locations))
        new_population.append(new_position)
    return new_population

def restore_valid_permutation(permutation, num_cities):
    """Restores a valid permutation by removing duplicates and restoring missing cities."""
    seen = set()
    new_permutation = []
    missing_cities = set(range(num_cities)) - set(permutation)
    for city in permutation:
        if city not in seen:
            new_permutation.append(city)
            seen.add(city)
        else:
            new_permutation.append(missing_cities.pop())
    return new_permutation

def find_best_neighbor(current_position, city_locations):
    """Finds the index of the best neighboring city in the current route."""
    best_neighbor_index = 0
    min_distance = float('inf')
    for i in range(len(current_position)):
        for j in range(len(current_position)):
            if i != j:
                distance_ij = distance(city_locations[current_position[i]], city_locations[current_position[j]])
                if distance_ij < min_distance:
                    min_distance = distance_ij
                    best_neighbor_index = j
    return best_neighbor_index

def select_best_individuals(population, fitness_values, num_individuals_to_select):
    """Selects the top 'num_individuals_to_select' individuals based on fitness."""
    sorted_individuals = sorted(zip(population, fitness_values), key=lambda x: x[1])
    return [individual for individual, _ in sorted_individuals[:num_individuals_to_select]]

def parse_tsplib(filename):
    """Parses a TSPLIB file to extract city coordinates."""
    problem = tsplib95.load(filename)
    city_locations = [(node[0], node[1]) for node in problem.node_coords.values()]
    print(f"Parsed {len(city_locations)} cities from {filename}.")
    return city_locations

def main():
    # Problem parameters
    filename = 'berlin52.tsp'  # Change this to the path of your TSPLIB file
    city_locations = parse_tsplib(filename)
    num_cities = len(city_locations)
    print(f"Number of cities: {num_cities}")

    # Flow Regime Algorithm parameters
    num_individuals = 20
    alpha = 0.5  # Probability of attraction to better solution
    beta = 0.3  # Probability of random disturbance
    gamma = 1.0  # Scaling factor for velocity
    num_generations = 100

    # Generate initial population
    population = generate_initial_population(num_individuals, num_cities)
    best_solution = None
    best_fitness = float('inf')

    # Main loop (iterations)
    for generation in range(num_generations):
        # Calculate fitness for each individual
        fitness_values = [calculate_fitness(permutation, city_locations) for permutation in population]

        # Select the best individuals
        population = select_best_individuals(population, fitness_values, num_individuals)

        # Update positions
        population = update_positions(population, city_locations, alpha, beta, gamma)

        # Calculate fitness for the updated population
        fitness_values = [calculate_fitness(permutation, city_locations) for permutation in population]

        # Update best solution
        for i in range(num_individuals):
            fitness = fitness_values[i]
            if fitness < best_fitness:
                best_solution = population[i]
                best_fitness = fitness

        # Print progress
        if generation % 10 == 0:
            print(f"Generation: {generation}, Best Distance: {best_fitness}")

    # Print final results
    print("Final Best Solution:", best_solution)
    print("Final Best Distance:", best_fitness)

if __name__ == "__main__":
    main()
