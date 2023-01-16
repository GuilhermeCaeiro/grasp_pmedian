import time
import numpy as np
from scipy.sparse.csgraph import floyd_warshall
import os
import random
import math
import copy
import operator
import matplotlib.pyplot as plt
import seaborn as sns

class Solution:
    def __init__(self, solution, fitness):
        self.solution = solution
        self.fitness = fitness

class GRASP:
    def __init__(self, instance, rcl_size, max_iterations, local_search_method, seed = 0):
        self.instance = instance
        self.max_iterations = max_iterations
        self.rcl_size = rcl_size
        self.local_search_method = local_search_method # Can be "first_improvement" or "best_improvement".
        #self.encoding = encoding
        self.best_solution = Solution(None, np.inf)
        self.solutions = []
        self.elite = []
        self.seed = seed

        random.seed(self.seed)
        np.random.seed(self.seed)

    def fitness_old(self, solution):
        p_costs = self.instance.costs[np.where(solution == 1)[0]]
        p_costs = sum(p_costs.min(axis = 0))

        penalty = 100 * abs(self.instance.p - sum(solution)[0])
        #print(p_costs, penalty)
        
        return p_costs + penalty

    def fitness(self, solution):
        p_costs = self.instance.costs[solution]
        p_costs = sum(p_costs.min(axis = 0))
        penalty = 100 * abs(self.instance.p - len(solution))
        #print(p_costs, penalty)
        
        return p_costs + penalty

    def fitness_change(self, original_solution, to_one, to_zero):
        pass

    def loop(self):
        start_time = time.time()

        for iteration in range(0, self.max_iterations):
            iteration_start_time = time.time()
            
            if (iteration % 1000) == 0:
                print("Iteration", iteration, "Time", time.time() - start_time)
            
            # greedy randomized search
            solution = self.greed_randomized_search(self.instance, self.rcl_size)
            #print("Initial solution", solution.fitness, sum(solution.solution))
            # local search
            solution = self.local_search(solution, self.local_search_method, self.instance)
            #print("Final solution", solution.fitness, sum(solution.solution))

            iteration_finish_time = time.time()
            #print("It.", iteration, "solution:", solution.fitness)
            
            if solution.fitness < self.best_solution.fitness:
                self.best_solution = solution
                
            self.solutions.append(solution)

        finish_time = time.time()
        
        solution_payload = {
            "seed": self.seed,
            "solution": self.best_solution,
            "elite": self.elite,
            "start_time": start_time,
            "finish_time": finish_time,
            "total_time": finish_time - start_time,
        }

        return solution_payload

    def greed_randomized_search(self, instance, rcl_size):
        start_time = time.time()
        candidate_locations = list(range(instance.n_candidate_locations))
        partial_solution = []
        sol_plus_fitness_times = []
        chosen = None

        for i in range(instance.p):
            candidate_solutions = []
            for candidate_location in candidate_locations:
                candidate_solution = partial_solution + [candidate_location]
                #candidate_solution = np.zeros((instance.n_candidate_locations, 1))
                #candidate_solution[partial] = 1
                stime = time.time()
                candidate_solution = Solution(candidate_solution, self.fitness(candidate_solution))
                sol_plus_fitness_times.append(time.time() - stime)
                candidate_solution.candidate_location = candidate_location # created a new attribute just to store this vale

                candidate_solutions.append(candidate_solution)

            current_rcl_size = math.ceil(len(candidate_solutions) * rcl_size)
            rcl = sorted(candidate_solutions, key=operator.attrgetter("fitness"), reverse=False)[:current_rcl_size]
            chosen = random.choice(rcl)
            
            candidate_locations.remove(chosen.candidate_location)
            partial_solution = list(np.where(chosen.solution == 1)[0])
        
        finish_time = time.time()
        print("GRS time", finish_time - start_time, "Total sol + fitness time", sum(sol_plus_fitness_times), "Average sol + fitness time", np.mean(sol_plus_fitness_times))
        return chosen

                

    def local_search(self, solution, local_search_method, instance):
        start_time = time.time()
        #ones = np.where(solution.solution == 1)[0]
        #zeros = np.where(solution.solution == 0)[0]
        chosen_locations = solution.solution
        not_chosen_locations = [element for element in list(range(instance.n_candidate_locations)) if element not in chosen_locations]
        improved_solution = solution

        for chosen in chosen_locations:
            for not_chosen in not_chosen_locations:
                candidate_solution = copy.deepcopy(solution.solution)
                #print(one, zero)
                #print(ones)
                #print(zeros)
                #print("\n\n\n")
                #print(candidate_solution)

                #candidate_solution[one] = 0
                #candidate_solution[zero] = 1
                candidate_solution.remove(chosen)
                candidate_solution.append(not_chosen)
                
                #print(candidate_solution)
                #exit()

                candidate_solution = Solution(candidate_solution, self.fitness(candidate_solution))

                if candidate_solution.fitness < improved_solution.fitness:
                    improved_solution = candidate_solution

                    if local_search_method == "first_improvement":
                        return improved_solution

        finish_time = time.time()
        print("LS time", finish_time - start_time)
        return improved_solution

    def plot_solution_distribution(self):
        plot = sns.distplot(
            [solution.fitness for solution in self.solutions], 
            hist=True, 
            kde=False, 
            bins=int(100), 
            color = 'blue',
            hist_kws={'edgecolor':'black'}
        )

        figure = plot.get_figure()
        figure.savefig("solution_distribution.png") 


class Instance:
    def __init__(self, path):
        self.file_path = path
        self.name = os.path.basename(self.file_path)
        self.p = None # number of facilities
        self.n_candidate_locations = None # number of candidate locations
        self.n_customer_locations = None # number of customer locations
        self.costs = None # cost matrix

        self.load_instance()

    def load_instance(self):
        lines = []

        with open(self.file_path, "r") as file:
            lines = file.readlines()

        for i in range(0, len(lines)):
            values = [int(value) for value in lines[i].strip().split(" ")]

            if i == 0: # first line
                self.n_candidate_locations = values[0]
                self.n_customer_locations = values[0]
                self.p = values[2]
                self.costs = np.ones((self.n_candidate_locations, self.n_customer_locations)) * np.inf
                np.fill_diagonal(self.costs, 0)
                print(self.costs)
            else:
                # accounting for the fact that np.array indexes start from 0
                values[0] -= 1
                values[1] -= 1
                # defining a bidirection edge
                self.costs[values[0], values[1]] = values[2]
                self.costs[values[1], values[0]] = values[2]

        np.savetxt(self.name + "_pre_floyd.csv", self.costs, delimiter = ',')
        print(self.costs)

        self.costs = floyd_warshall(self.costs)

        np.savetxt(self.name + "_post_floyd.csv", self.costs, delimiter = ',')
        print(self.costs)




grasp = GRASP(Instance("pmed1.txt"), 0.5, 10000, "best_improvement")
results = grasp.loop()
print(results["solution"].fitness)
grasp.plot_solution_distribution()