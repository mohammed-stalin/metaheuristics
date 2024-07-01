import random
import math
import tsplib95

def distance(city1, city2):
    """Calculates the Euclidean distance between two cities."""
    x1, y1 = city1
    x2, y2 = city2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def generate_initial_population(num_falcons, num_cities):
    """Creates a population of falcon positions (candidate solutions)."""
    population = []
    for _ in range(num_falcons):
        permutation = list(range(num_cities))
        random.shuffle(permutation)
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

def levy_flight_step(beta):
    """Generates a Levy flight step size."""
    return random.gammavariate(1, beta)

def two_opt_swap(route, i, k):
    """Performs a 2-opt swap on the route."""
    new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
    return new_route

def local_search(route, city_locations):
    """Improves the route using 2-opt local search."""
    best_route = route
    best_distance = calculate_fitness(route, city_locations)
    for i in range(len(route) - 1):
        for k in range(i + 1, len(route)):
            new_route = two_opt_swap(route, i, k)
            new_distance = calculate_fitness(new_route, city_locations)
            if new_distance < best_distance:
                best_route = new_route
                best_distance = new_distance
    return best_route

def update_falcon_positions(population, city_locations, beta, p_a, p_d):
    """Updates falcon positions based on exploration, exploitation, and agility."""
    new_population = []
    for falcon in population:
        new_position = falcon.copy()
        if random.random() < p_a:
            # Exploitation: 2-opt local search
            new_position = local_search(new_position, city_locations)
        elif random.random() < p_a + p_d:
            # Agility: Random swap
            i, k = random.sample(range(len(falcon)), 2)
            new_position[i], new_position[k] = new_position[k], new_position[i]
        else:
            # Exploration: Levy flight
            step_size = int(levy_flight_step(beta) * len(falcon))
            i = random.randint(0, len(falcon) - 1)
            new_position = new_position[i:] + new_position[:i]
            new_position = new_position[:step_size] + new_position[step_size:][::-1]
        
        new_population.append(new_position)
        print(f"Updated falcon position: {new_position[:10]}...")  # Print first 10 for brevity
    return new_population

def select_best_falcons(population, fitness_values, num_falcons_to_select):
    """Selects the top 'num_falcons_to_select' falcons based on fitness."""
    sorted_falcons = sorted(zip(population, fitness_values), key=lambda x: x[1])
    return [falcon for falcon, _ in sorted_falcons[:num_falcons_to_select]]

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

    # FOA parameters (adjust these as needed)
    num_falcons = 50
    beta = 2  # Levy flight parameter for exploration
    p_a = 0.7  # Probability of exploitation
    p_d = 0.4  # Probability of agility

    # Main loop (iterations)
    max_iterations = 1000
    best_solution = None
    best_fitness = float('inf')
    population = generate_initial_population(num_falcons, num_cities)
    print(f"Generated initial population: {population[:3]}...")  # Print first 3 for brevity

    for iteration in range(max_iterations):
        # Calculate fitness for each falcon
        fitness_values = [calculate_fitness(permutation, city_locations) for permutation in population]
        print(f"Iteration {iteration}: Fitness values: {fitness_values[:3]}...")  # Print first 3 for brevity

        # Update best solution before updating positions
        for i in range(num_falcons):
            fitness = fitness_values[i]
            if fitness < best_fitness:
                best_solution = population[i]
                best_fitness = fitness

        # Update falcon positions
        population = update_falcon_positions(population, city_locations, beta, p_a, p_d)

        # Recalculate fitness after updating positions
        fitness_values = [calculate_fitness(permutation, city_locations) for permutation in population]

        # Update best solution after updating positions
        for i in range(num_falcons):
            fitness = fitness_values[i]
            if fitness < best_fitness:
                best_solution = population[i]
                best_fitness = fitness

        # Print progress (optional)
        if iteration % 10 == 0:
            print(f"Iteration: {iteration}, Best Distance: {best_fitness}")

    # Print final results
    print("Final Best Solution:", best_solution)
    print("Final Best Distance:", best_fitness)

if __name__ == "__main__":
    main()
