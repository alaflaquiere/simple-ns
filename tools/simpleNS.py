import os
import time
import multiprocessing as mp
import _pickle as pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, jensenshannon


# indexation
I_PARENT = 0  # index of the parent in the previous generation
I_GENERATION = 1  # generation at which the agent is created
I_SELECTED_POP = 2  # flag indicating if the individuals is selected to be part of the next population
I_SELECTED_ARC = 3  # flag indicating if the individuals is selected to be part of the archive
I_AGE = 4  # age of the individuals = number of generations since its creation
I_GENOME = [5, 6]  # 2D genome
I_BEHAVIOR = 7  # 1D behavior
I_DISTANCE = 8  # distance to the parent when the individual is created
I_NOVELTY = 9  # novelty of the individual
I_COVERAGE = 10  # coverage associated with the genome
I_UNIFORMITY = 11  # uniformity associated with the genome
I_CUM_COVERAGE = 12  # cumulative coverage associated with the current group of individuals
I_CUM_UNIFORMITY = 13  # cumulative uniformity associated with the current group of individuals
SIZE_INDIVIDUAL = 14  # length of the descriptor for each individual


class Experiment:
    """
    Novelty search in a simple simulated setting where a 2D genome is mapped to a 1D behavior space via a simple  non-linear mapping.
    """

    def __init__(self, params, display=False):
        """Constructor

        Args:
            params: dictionary with the following keys
                mapping (str): type of genome-to-behavior mapping {'linear', 'hyperbolic', 'bounded_linear', 'linear_seesaw', 'multiplicative',
                                                                   'soft_multiplicative', 'hyperbolic_seesaw', 'multiplicative_seesaw', 'cosinus',
                                                                   '2D_cosinus', 'multiplicative_cosinus', 'peaks', '2D_peaks'}
                eta (float): mutation spread parameter
                n_pop (int): number of individuals in the population
                n_offspring (int): number of offsprings generated from the population at each generation
                criterion (str): type of novelty computation {'novelty', 'hull', 'fitness', 'random'}
                n_neighbors (int): number of closest neighbors to compute the novelty ("novelty" criterion)
                best_fit (float): arbitrary behavior with the maximum fitness ("fitness" criterion)
                n_selected (int): number of offsprings added to the archive at each generation
                n_evolvability (int): number of samples generated from each genome to evaluate its evolvability
                n_epochs (int): number of generations of the search process
                addition (str): strategy to add individuals to the archive {'novelty', 'random'}
                restart (int): generation at which the population is re-initialized
                frozen (int): generation at which the reference set for the novelty computation is frozen
            display (bool): display the state of the search at each generation

        """
        assert params["mapping"] in ['linear', 'hyperbolic', 'bounded_linear', 'linear_seesaw', 'multiplicative',
                                     'soft_multiplicative', 'hyperbolic_seesaw', 'multiplicative_seesaw', 'cosinus',
                                     '2D_cosinus', 'multiplicative_cosinus', 'peaks', '2D_peaks'], "incorrect type of mapping"
        assert params["eta"] > 0, "eta must be greater than 0"
        assert params["n_pop"] > 0, "n_pop must be greater than 0"
        assert params["n_offspring"] > 0, "n_offspring must be greater than 0"
        assert params["criterion"] in ['novelty', 'hull', 'fitness', 'random'], "incorrect selection criterion"
        assert params["n_neighbors"] > 0, "n_neighbors must be greater than 0"
        assert params["n_selected"] > 0, "n_selected must be greater than 0"
        assert params["n_evolvability"] > 0, "n_evolvability must be greater than 0"
        assert params["n_epochs"] > 0, "n_epochs must be greater than 0"
        assert params["addition"] in ['novelty', 'random'], "incorrect addition criterion"
        assert params["restart"] > 0, "restart must be greater than 0"
        assert params["frozen"] > 0, "frozen must be greater than 0"

        # experiment parameters
        self.mapping = params["mapping"]
        self.eta = params["eta"]
        self.n_pop = params["n_pop"]
        self.n_offspring = params["n_offspring"]
        self.criterion = params["criterion"]
        self.n_neighbors = params["n_neighbors"]
        self.best_fit = params["best_fit"]
        self.n_selected = params["n_selected"]
        self.n_evolvability = params["n_evolvability"]
        self.n_epochs = params["n_epochs"]
        self.addition = params["addition"]
        self.restart = np.inf if params["restart"] is None else params["restart"]
        self.frozen = np.inf if params["frozen"] is None else params["frozen"]

        self.display = display

        # preallocate memory - individuals are stored as a vector of float, which is faster in access than storing them as a class instances
        self.archive = np.full((self.n_epochs + 1, self.n_selected, SIZE_INDIVIDUAL), np.nan, dtype=np.float32)
        self.population = np.full((self.n_epochs + 1, self.n_pop, SIZE_INDIVIDUAL), np.nan, dtype=np.float32)
        self.offsprings = np.full((self.n_epochs, self.n_offspring, SIZE_INDIVIDUAL), np.nan, dtype=np.float32)

        # iterator
        self.t = 0

    def gene_to_behavior(self, g):
        """Non-linear mapping from genome to behavior in [-5, 5]

        Args:
            g (np.array((N, d), float)): genomes

        Returns:
            np.array((N), float): behaviors

        """
        assert all(np.abs(g.flatten()) <= 5.), "the gene values should be in [-5., 5.]"

        if self.mapping == "linear":
            behavior = g[:, 0]
        elif self.mapping == "hyperbolic":
            behavior = g[:, 0] ** 2 / 2.5 - 5
        elif self.mapping == "bounded_linear":
            behavior = np.maximum(np.minimum(g[:, 0], np.ones_like(g[:, 0])), -np.ones_like(g[:, 0]))
        elif self.mapping == "linear_seesaw":
            behavior = 10 * np.abs((g[:, 0] % 2) - 1) - 5
        elif self.mapping == "multiplicative":
            behavior = g[:, 0] * g[:, 1] / 5
        elif self.mapping == "soft_multiplicative":
            behavior = 10 / np.pi * np.arctan(g[:, 0] * g[:, 1])
        elif self.mapping == "hyperbolic_seesaw":
            behavior = 10 * np.abs(((g[:, 0] ** 2) % 2) - 1) - 5
        elif self.mapping == "multiplicative_seesaw":
            behavior = 10 * np.abs(((10 * g[:, 0] * g[:, 1]) % 2) - 1) - 5
        elif self.mapping == "cosinus":
            behavior = 5 * np.cos(0.4 * g[:, 0] ** 2)
            # behavior = 5 * np.cos(np.pi / 2 + np.sign(g[:, 0]) * 0.4 * g[:, 0] ** 2)
        elif self.mapping == "2D_cosinus":
            behavior = 5 * np.cos(np.pi / 2 + 2 * np.pi * g[:, 0] / (11 - 1.8 * g[:, 1]))
        elif self.mapping == "multiplicative_cosinus":
            behavior = 5 * np.cos(g[:, 0] * g[:, 1])
        elif self.mapping == "peaks":
            behavior = (np.exp(-(g[:, 0] + 4) ** 2 / 0.01) + np.exp(-(g[:, 0] + 2) ** 2 / 0.1) + np.exp(-(g[:, 0] - 2) ** 2 / 1)) * 10 - 5
        elif self.mapping == "2D_peaks":
            behavior = ((1 - np.exp(-((g[:, 0] + 3) ** 2 + (g[:, 1] - 3) ** 2) / 0.1)) * (1 - np.exp(-((g[:, 0] - 3) ** 2 + (g[:, 1] + 3) ** 2) / 0.5)) *
                        (1 - np.exp(-((g[:, 0] + 3) ** 2 + (g[:, 1] + 3) ** 2) / 1.0)) * (
                                    1 - np.exp(-((g[:, 0] - 3) ** 2 + (g[:, 1] - 3) ** 2) / 5.0))) * 10 - 5
        else:
            behavior = 0 * g[:, 0]
        return behavior

    def mutate_genome(self, g, low=-5., high=5.):
        """Mutation operator

        Args:
            g (np.array((N, d), float)): genomes
            low (float): lower bound for mutated genes
            high (float): higher bound for mutated genes

        Returns:
            new_genomes (np.array((N, d), float)): mutated genomes

        """
        if low > high:
            low, high = high, low

        mut_power = 1.0 / (self.eta + 1.)

        rands = np.random.rand(*g.shape)
        mask = rands < 0.5

        xy = np.full_like(g, np.nan)
        xy[mask] = 1.0 - ((g[mask] - low) / (high - low))
        xy[~mask] = 1.0 - ((high - g[~mask]) / (high - low))

        val = np.full_like(g, np.nan)
        val[mask] = 2.0 * rands[mask] + (1.0 - 2.0 * rands[mask]) * xy[mask] ** (self.eta + 1)
        val[~mask] = 2.0 * (1.0 - rands[~mask]) + 2.0 * (rands[~mask] - 0.5) * xy[~mask] ** (self.eta + 1)

        delta_q = np.full_like(g, np.nan)
        delta_q[mask] = val[mask] ** mut_power - 1.0
        delta_q[~mask] = 1.0 - val[~mask] ** mut_power

        new_genomes = g + delta_q * (high - low)
        new_genomes = np.minimum(np.maximum(new_genomes, low * np.ones_like(new_genomes)), high * np.ones_like(new_genomes))

        return new_genomes

    def compute_novelty(self, new_b, old_b=None):
        """Compute the novelty of new behaviors, compared to a pool of new + old behaviors.

        Different strategies are possible to compute the novelty:
        - "novelty": novelty is the average distance to the n_neighbors closest neighbors
        - "hull": novelty is the smallest distance to the hull of old_b ([min(old_b), max(old_b)])
        - "fitness": novelty is the distance to the best_fit target
        - "random": novelty is randomly assigned from a unitary uniform distribution

        Args:
            new_b (np.array((N), float)): behaviors to compute the novelty for
            old_b (np.array((N), float)): archive of behaviors used as reference to compute the novelty

        Returns:
            novelties (np.array((N), float)): novelties of the new behaviors

        """
        if self.criterion not in ["novelty", "hull", "fitness", "random"]:
            raise ValueError("criterion can only be 'novelty', 'hull', 'fitness', or 'random'")

        if self.criterion == "novelty":
            if old_b is None:
                old_b = self.get_reference_behaviors()
            distances = cdist(new_b.reshape(-1, 1), old_b.reshape(-1, 1))
            distances = np.sort(distances, axis=1)
            novelties = np.mean(distances[:, :self.n_neighbors], axis=1)  # Note: one distance might be 0 (distance to self)

        elif self.criterion == "hull":
            if old_b is None:
                old_b = self.get_reference_behaviors()
            hull_min = np.min(old_b)
            hull_max = np.max(old_b)
            smallest_distances = np.maximum(hull_min - new_b, new_b - hull_max)
            novelties = np.maximum(smallest_distances, np.zeros_like(smallest_distances))

        elif self.criterion == "fitness":
            novelties = -np.abs(new_b - self.best_fit)  # distance to the best fit

        elif self.criterion == "random":
            novelties = np.random.rand(len(new_b))

        else:
            novelties = []

        return novelties

    def evaluate_coverage_and_uniformity(self, g, n_bins=50):
        """Evaluate the coverage and uniformity of genome(s) via sampling.

        Args:
            g (np.array((N, d), float)): genomes
            n_bins (int): number of bins in the behavior space

        Returns:
            coverages (np.array((N), float)): ratios of bins covered by sampling each genomes
            uniformities (np.array((N), float)): uniformities of the sampling from each genomes
            cum_coverage (float): ratios of bins covered by sampling all genomes
            cum_uniformity (float): uniformity of the sampling from all genomes

        """
        num = g.shape[0]
        coverages = np.zeros(num)
        uniformities = np.zeros(num)

        # for each individual
        all_behavior_samples = np.zeros(num * self.n_evolvability)
        for i in range(num):
            genomes_samples = self.mutate_genome(np.tile(g[[i], :], (self.n_evolvability, 1)))
            behavior_samples = self.gene_to_behavior(genomes_samples)

            hist, _ = np.histogram(behavior_samples, np.linspace(-5, 5, n_bins + 1))
            hist = hist[np.nonzero(hist)] / self.n_evolvability
            hist_uniform = np.mean(hist) * np.ones_like(hist)

            all_behavior_samples[i * self.n_evolvability: (i + 1) * self.n_evolvability] = behavior_samples
            coverages[i] = len(hist) / n_bins
            uniformities[i] = 1 - jensenshannon(hist, hist_uniform, base=2)

        # cumulative over all genomes
        cum_hist, _ = np.histogram(all_behavior_samples, np.linspace(-5, 5, n_bins + 1))
        cum_hist = cum_hist[np.nonzero(cum_hist)] / (num * self.n_evolvability)
        cum_hist_uni = np.mean(cum_hist) * np.ones_like(cum_hist)

        cum_coverage = len(cum_hist) / n_bins
        cum_uniformity = 1 - jensenshannon(cum_hist, cum_hist_uni, base=2)

        return coverages, uniformities, cum_coverage, cum_uniformity

    def initialize_population(self):
        """Generates an initial population of individuals at generation self.t.

        """
        self.population[self.t, :, I_PARENT] = np.full(self.n_pop, np.nan)
        self.population[self.t, :, I_GENERATION] = -np.ones(self.n_pop, dtype=np.float32)
        self.population[self.t, :, I_SELECTED_POP] = np.zeros(self.n_pop, dtype=np.float32)
        self.population[self.t, :, I_SELECTED_ARC] = np.zeros(self.n_pop, dtype=np.float32)
        self.population[self.t, :, I_AGE] = np.zeros(self.n_pop, dtype=np.float32)
        self.population[self.t, :, :][:, I_GENOME] = np.random.rand(self.n_pop, 2) - 0.5
        self.population[self.t, :, I_BEHAVIOR] = self.gene_to_behavior(self.population[self.t, :, :][:, I_GENOME])
        self.population[self.t, :, I_DISTANCE] = np.zeros(self.n_pop, dtype=np.float32)
        self.population[self.t, :, I_NOVELTY] = self.compute_novelty(self.population[self.t, :, I_BEHAVIOR],
                                                                     old_b=self.population[self.t, :, I_BEHAVIOR])  # compare only to itself
        self.population[self.t, :, I_COVERAGE], self.population[self.t, :, I_UNIFORMITY],\
            self.population[self.t, :, I_CUM_COVERAGE], self.population[self.t, :, I_CUM_UNIFORMITY] = \
            self.evaluate_coverage_and_uniformity(self.population[self.t, :, :][:, I_GENOME])

    def initialize_archive(self):
        """Add most novel individuals for the generation self.t to the archive.

        """
        # add the best individuals to the archive
        selected, self.archive[self.t, :, :] = self.select_individuals(self.population[self.t, :, :], self.n_selected, "novelty")
        self.population[self.t, selected, I_SELECTED_ARC] = 1.

    def generate_offsprings(self):
        """Generates offsprings from the population at generation self.t.

        """
        self.offsprings[self.t, :, I_GENERATION] = float(self.t)
        self.offsprings[self.t, :, I_PARENT] = np.random.randint(0, self.n_pop, self.n_offspring)
        self.offsprings[self.t, :, I_SELECTED_POP] = np.zeros(self.n_offspring, dtype=np.float32)
        self.offsprings[self.t, :, I_SELECTED_ARC] = np.zeros(self.n_offspring, dtype=np.float32)
        self.offsprings[self.t, :, I_AGE] = np.zeros(self.n_offspring, dtype=np.float32)
        self.offsprings[self.t, :, :][:, I_GENOME] = \
            self.mutate_genome(self.population[self.t, self.offsprings[self.t, :, I_PARENT].astype(int), :][:, I_GENOME])
        self.offsprings[self.t, :, I_BEHAVIOR] = self.gene_to_behavior(self.offsprings[self.t, :, :][:, I_GENOME])
        self.offsprings[self.t, :, I_DISTANCE] = np.abs(self.offsprings[self.t, :, I_BEHAVIOR]
                                                        - self.population[self.t, self.offsprings[self.t, :, I_PARENT].astype(int), I_BEHAVIOR])
        old_behaviors = self.get_reference_behaviors()
        self.offsprings[self.t, :, I_NOVELTY] = self.compute_novelty(self.offsprings[self.t, :, I_BEHAVIOR], old_behaviors)
        self.offsprings[self.t, :, I_COVERAGE], self.offsprings[self.t, :, I_UNIFORMITY],\
            self.offsprings[self.t, :, I_CUM_COVERAGE], self.offsprings[self.t, :, I_CUM_UNIFORMITY] = \
            self.evaluate_coverage_and_uniformity(self.offsprings[self.t, :, :][:, I_GENOME])

    @staticmethod
    def select_individuals(individuals, n, strategy):
        """Selects n individuals according to the strategy.

        Args:
            individuals (np.array((N, d), float)): set of individuals
            n (in): number of individuals to select
            strategy (str): strategy to select the individuals - can be 'novelty' or 'random'

        Returns:
            selected (list(int)): index of the selected individuals
            (np.array((n, d), float)): set of selected individuals

        """
        if strategy == "novelty":
            selected = np.argsort(individuals[:, I_NOVELTY])[-n:]
        elif strategy == "random":
            selected = np.random.choice(individuals.shape[0], n, replace=False)
        else:
            selected = []
        return selected, individuals[selected, :]

    def select_and_add_to_archive(self):
        selected, self.archive[self.t + 1, :, :] = self.select_individuals(self.offsprings[self.t, :, :], self.n_selected, self.addition)
        self.offsprings[self.t, selected, I_SELECTED_ARC] = 1.

    def get_reference_behaviors(self):
        """Concatenates the behaviors from the archive, population, and offsprings from generation t or the frozen generation.

        Returns:
            (np.array((N,d), float)): set of behaviors

        """
        return np.hstack((self.population[min(self.t, self.frozen), :, I_BEHAVIOR],
                          self.offsprings[min(self.t, self.frozen), :, I_BEHAVIOR],
                          self.archive[:(min(self.t, self.frozen) + 1), :, I_BEHAVIOR].reshape(-1)))

    def create_next_generation(self):
        extended_population = np.vstack((self.population[self.t, :, :], self.offsprings[self.t, :, :]))
        selected, self.population[self.t + 1, :, :] = self.select_individuals(extended_population, self.n_pop, "novelty")
        selected_population = selected[selected < self.n_pop]
        selected_offsprings = selected[selected >= self.n_pop] - self.n_pop
        self.population[self.t, selected_population, I_SELECTED_POP] = 1.
        self.offsprings[self.t, selected_offsprings, I_SELECTED_POP] = 1.
        self.population[self.t + 1, :, I_AGE] += 1.
        self.population[self.t + 1, :, I_SELECTED_POP] = 0.  # erase possible heritage from previous generation
        self.population[self.t + 1, :, I_SELECTED_ARC] = 0.  # erase possible heritage from previous generation
        _, _, self.population[self.t + 1, :, I_CUM_COVERAGE], self.population[self.t + 1, :, I_CUM_UNIFORMITY] = \
            self.evaluate_coverage_and_uniformity(self.population[self.t + 1, :, :][:, I_GENOME])  # update the cumulative coverage and uniformity

    def run_novelty_search(self):
        """Applies the Novelty Search algorithm."""

        # initialize the run - create a random population and add the best individuals to the archive
        self.initialize_population()
        self.initialize_archive()

        # iterate
        while self.t < self.n_epochs:

            # in case of a restart, reinitialize the population and start again
            if self.t == self.restart:
                self.initialize_population()

            # generate offsprings from random parents in the population
            self.generate_offsprings()

            # update the population novelty
            self.population[self.t, :, I_NOVELTY] = self.compute_novelty(self.population[self.t, :, I_BEHAVIOR])

            # add the most novel offsprings to the archive
            self.select_and_add_to_archive()

            # keep the most novel individuals in the (population + offsprings) in the population
            self.create_next_generation()

            if self.display:
                self.display_generation()

            self.t += 1

    def display_generation(self):
        """Displays the state of the search at generation t.

        """
        if self.t == 0:
            plt.subplot(121)
            plt.subplot(122)

        plt.sca(plt.gcf().axes[0])
        plt.cla()
        plt.plot(self.archive[:self.t + 1, :, I_GENOME[0]].reshape(-1), self.archive[:self.t + 1, :, I_GENOME[1]].reshape(-1), "k.", label="archive")
        plt.plot(self.population[self.t, :, I_GENOME[0]], self.population[self.t, :, I_GENOME[1]], "b.", label="population")
        plt.plot(self.offsprings[self.t, :, I_GENOME[0]], self.offsprings[self.t, :, I_GENOME[1]], "r.", label="offsprings")
        plt.axis([-5, 5, -5, 5])
        if self.t == 0:
            plt.legend()

        plt.sca(plt.gcf().axes[1])
        plt.plot(self.archive[:self.t + 1, :, I_BEHAVIOR].reshape(-1), np.full(((self.t + 1) * self.n_selected), self.t),
                 "k.", markersize=10, label="archive")
        plt.plot(self.population[self.t, :, I_BEHAVIOR], np.full(self.n_pop, self.t + 0.1),
                 "b.", markersize=10, label="population")
        plt.plot(self.offsprings[self.t, :, I_BEHAVIOR], np.full(self.n_offspring, self.t + 0.2),
                 "r.", markersize=10, label="offsprings")

        novelty_landscape = self.compute_novelty(np.linspace(-5, 5, 100))
        plt.plot(np.linspace(-5, 5, 100), novelty_landscape + self.t, color="g", label="novelty")

        selected = np.argwhere(self.population[self.t, :, I_SELECTED_POP] == 1.)
        plt.plot(self.population[self.t, selected, I_BEHAVIOR], np.full(len(selected), self.t + 0.1),
                 "bo", markersize=14, fillstyle="none", label="selected for population")
        selected = np.argwhere(self.offsprings[self.t, :, I_SELECTED_POP] == 1.)
        plt.plot(self.offsprings[self.t, selected, I_BEHAVIOR], np.full(len(selected), self.t + 0.2),
                 "bo", markersize=14, fillstyle="none")
        selected = np.argwhere(self.offsprings[self.t, :, I_SELECTED_ARC] == 1.)
        plt.plot(self.offsprings[self.t, selected, I_BEHAVIOR], np.full(len(selected), self.t + 0.2),
                 "kd", markersize=14, fillstyle="none", label="selected for population")
        plt.xlim([-5, 5])
        if self.t == 0:
            plt.legend()

        plt.title("generation " + str(self.t))
        plt.show(block=False)
        plt.waitforbuttonpress()

    def get_results(self):
        """Yields the archive history, population history, and offsprings history in a dictionary.

        Returns:
            d (dict)

        """
        d = {"archive": self.archive,
             "population": self.population,
             "offsprings": self.offsprings}
        return d


def create_and_run_experiment(params, display=False):
    """Creates a Novelty Search algorithm and run the search according to the input parameters.
       It is possible to display the evolution of the search.

    Args:
        params (dict): parameters of the search
        display (bool): flag to display each generation during the run

    Returns:
        data (dict): dictionary containing the archive history, population history, and offsprings history

    """
    my_exp = Experiment(params, display)
    my_exp.run_novelty_search()
    data = my_exp.get_results()
    return data


def run_in_parallel(params):
    """Distributes the runs over multiple CPU - the visualization of the search is disabled.

    Args:
        params (dict): parameters of the search

    Returns:
        runs_data (list(dict)): list of dictionaries containing the archive history, population history, and offsprings history of each run

    """
    print("running {} simulations in parallel on {} workers".format(params["n_runs"], max(1, os.cpu_count() - 1)), flush=True)
    time.sleep(0.1)  # prevents the progress bar to (strangely) appear before the last print statement
    tic = time.time()
    progress_bar = tqdm(total=params["n_runs"])
    with mp.Pool(max(1, os.cpu_count() - 1)) as pool:
        runs_data = [pool.apply_async(create_and_run_experiment, args=(params,), callback=lambda _: progress_bar.update(1)) for _ in range(params["n_runs"])]
        runs_data = [r.get() for r in runs_data]
    print("elapsed time: {:.2f} minutes".format((time.time() - tic) / 60))
    progress_bar.close()
    return runs_data


def save_experiment_results(params, d):
    """Save experiments results.

    Args:
        params (dict): experiment parameters
        d: (list(dict)): results from experiments

    Returns:
        filename (str): path to the saved file
    """
    dic = {"parameters": params,
           "results": d}

    if not os.path.exists("results"):
        os.mkdir("results")
        print("created folder: results")

    filename = time.strftime("%Y%m%d-%H%M%S") if params["name"] is None else params["name"]

    path = "results/" + filename + ".pkl"
    with open(path, "wb") as f:
        pickle.dump(dic, f)

    print("results saved in {}".format(path))

    return path
