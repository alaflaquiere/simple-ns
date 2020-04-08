import tkinter as tk
from tools.simpleNS import *

plt.ion()


class Analyzer:
    """Analyzer for results generated using simpleNS."""

    def __init__(self, file_path="", save_figures=True):
        """Constructor

        Args:
            file_path (str): path to the results file
            save_figures (bool): flag to save the figures in ./figures when they are displayed

        """
        if file_path == "":
            tk.Tk().withdraw()
            file_path = tk.filedialog.askopenfilename(initialdir="./results")

        print("analyzing the results from {}".format(file_path))
        self.experiment_path = file_path
        self.experiment_name, _ = os.path.splitext(os.path.basename(file_path))
        self.parameters, self.all_archives, self.all_populations, self.all_offsprings = self.load_experiment_results()
        self.n_epochs = self.parameters["n_epochs"]
        self.n_runs = self.parameters["n_runs"]
        self.n_pop = self.parameters["n_pop"]
        self.n_selected = self.parameters["n_selected"]
        self.n_offspring = self.parameters["n_offspring"]
        self.dummy_experiment = Experiment(self.parameters, display=False)
        self.all_stats = self.get_all_stats(group="population")
        self.save_figures = save_figures
        if self.save_figures and not os.path.exists("figures"):
            os.mkdir("figures")
            print("created folder: figures")
        if self.save_figures and not os.path.exists("figures/{}".format(self.experiment_name)):
            os.mkdir("figures/{}".format(self.experiment_name))
            print("created folder: figures/{}".format(self.experiment_name))

    def load_experiment_results(self):
        """Load the results of the experiment."""

        with open(self.experiment_path, "rb") as f:
            d = pickle.load(f)

        arch_collection = np.array([r["archive"] for r in d["results"]])
        pop_collection = np.array([r["population"] for r in d["results"]])
        off_collection = np.array([r["offsprings"] for r in d["results"]])
        return d["parameters"], arch_collection, pop_collection, off_collection

    @staticmethod
    def get_median_fq_ts(data):
        """Compute the median, first quartile, and third quartile for each run
           and across runs (median, first quartile, and third quartile of the medians of the runs)

        Args:
            data (np.array((N_r, N_g, N_i), float)): set of history of measures for individuals and for all the runs

        Returns:
            m (np.array((N_r, N_g), float)): median for each run
            fq (np.array((N_r, N_g), float)): first quartile for each run
            tq (np.array((N_r, N_g), float)): third quartile for each run
            m_m (np.array(N_g, float)): median of the medians across runs
            fq_m (np.array(N_g, float)): first quartile of the medians across runs
            tq_m (np.array(N_g, float)): third quartile of the medians across runs

        """
        m = np.median(data, axis=2)
        fq = np.quantile(data, 0.25, axis=2)
        tq = np.quantile(data, 0.75, axis=2)
        # median and quartiles over the median ind_coverages
        m_m = np.median(m, axis=0)
        fq_m = np.quantile(m, 0.25, axis=0)
        tq_m = np.quantile(m, 0.75, axis=0)
        return m, fq, tq, m_m, fq_m, tq_m

    def get_all_stats(self, group="population"):
        """Compute stats through generations for the input set of individuals.

        Args:
            group (str): group of individuals to analyze - can be 'population', 'archive', or 'offsprings'

        Returns:
            stats (dist): dictionary of all the computed stats.

        """
        assert group in ["population", "archive", "offsprings"], "incorrect group - should be 'population', 'archive', or 'offsprings'"
        if group == "population":
            d = self.all_populations
        elif group == "archive":
            d = self.all_archives
        elif group == "offsprings":
            d = self.all_offsprings
        else:
            return

        median_ind_coverages, fq_ind_coverages, tq_ind_coverages, median_median_ind_coverages, \
            fq_median_ind_coverages, tq_median_ind_coverages = self.get_median_fq_ts(d[:, :, :, I_COVERAGE])

        median_ind_uniformities, fq_ind_uniformities, tq_ind_uniformities, median_median_ind_uniformities, \
            fq_median_ind_uniformities, tq_median_ind_uniformities = self.get_median_fq_ts(d[:, :, :, I_UNIFORMITY])

        median_cum_coverages, fq_cum_coverages, tq_cum_coverages,\
            median_median_cum_coverages, fq_median_cum_coverages, tq_median_cum_coverages = self.get_median_fq_ts(d[:, :, :, I_CUM_COVERAGE])

        median_cum_uniformities, fq_cum_uniformities, tq_cum_uniformities,\
            median_median_cum_uniformities, fq_median_cum_uniformities, tq_median_cum_uniformities = self.get_median_fq_ts(d[:, :, :, I_CUM_UNIFORMITY])

        median_ages, fq_age, tq_age, median_median_ages, fq_median_ages, tq_median_ages = self.get_median_fq_ts(self.all_populations[:, :, :, I_AGE])

        median_distances, fq_distances, tq_distances,\
            median_median_distances, fq_median_distances, tq_median_distances = self.get_median_fq_ts(d[:, :, :, I_DISTANCE])

        stats = {"median_ind_coverages": median_ind_coverages,
                 "fq_ind_coverages": fq_ind_coverages,
                 "tq_ind_coverages": tq_ind_coverages,
                 "median_median_ind_coverages": median_median_ind_coverages,
                 "fq_median_ind_coverages": fq_median_ind_coverages,
                 "tq_median_ind_coverages": tq_median_ind_coverages,
                 "median_ind_uniformities": median_ind_uniformities,
                 "fq_ind_uniformities": fq_ind_uniformities,
                 "tq_ind_uniformities": tq_ind_uniformities,
                 "median_median_ind_uniformities": median_median_ind_uniformities,
                 "fq_median_ind_uniformities": fq_median_ind_uniformities,
                 "tq_median_ind_uniformities": tq_median_ind_uniformities,
                 "median_cum_coverages": median_cum_coverages,
                 "fq_cum_coverages": fq_cum_coverages,
                 "tq_cum_coverages": tq_cum_coverages,
                 "median_median_cum_coverages": median_median_cum_coverages,
                 "fq_median_cum_coverages": fq_median_cum_coverages,
                 "tq_median_cum_coverages": tq_median_cum_coverages,
                 "median_cum_uniformities": median_cum_uniformities,
                 "fq_cum_uniformities": fq_cum_uniformities,
                 "tq_cum_uniformities": tq_cum_uniformities,
                 "median_median_cum_uniformities": median_median_cum_uniformities,
                 "fq_median_cum_uniformities": fq_median_cum_uniformities,
                 "tq_median_cum_uniformities": tq_median_cum_uniformities,
                 "median_ages": median_ages,
                 "fq_age": fq_age,
                 "tq_age": tq_age,
                 "median_median_ages": median_median_ages,
                 "fq_median_ages": fq_median_ages,
                 "tq_median_ages": tq_median_ages,
                 "median_distances": median_distances,
                 "fq_distances": fq_distances,
                 "tq_distances": tq_distances,
                 "median_median_distances": median_median_distances,
                 "fq_median_distances": fq_median_distances,
                 "tq_median_distances": tq_median_distances
                 }

        return stats

    def display_mapping(self, reso=100):
        """Displays the genome-to-behavior mapping.

        Args:
            reso (int): resolution

        """
        xx, yy = np.meshgrid(np.linspace(-5, 5, reso), np.linspace(-5, 5, reso))
        genomes = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1)))
        behaviors = self.dummy_experiment.gene_to_behavior(genomes)
        plt.figure(figsize=(6, 6))
        plt.contourf(genomes[:, 0].reshape(reso, reso),
                     genomes[:, 1].reshape(reso, reso),
                     behaviors.reshape(reso, reso),
                     reso, cmap="cividis")
        plt.title("behavior")
        plt.xlabel("gene 1")
        plt.ylabel("gene 2")
        plt.axis("equal")
        plt.show()
        if self.save_figures:
            plt.savefig("figures/{}/gene-to-behavior mapping.svg".format(self.experiment_name))
            plt.savefig("figures/{}/gene-to-behavior mapping.png".format(self.experiment_name))

    def display_mapping_evolvability(self, reso=100):
        """Displays the evolvability (coverage + uniformity) of the genome-to-behavior mapping.

        Args:
            reso (int): resolution

        """
        xx, yy = np.meshgrid(np.linspace(-5, 5, reso), np.linspace(-5, 5, reso))
        genomes = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1)))
        coverages, uniformities, _, _ = self.dummy_experiment.evaluate_coverage_and_uniformity(genomes)

        plt.figure(figsize=(13, 6))
        plt.subplot(121)
        plt.contourf(genomes[:, 0].reshape(reso, reso),
                     genomes[:, 1].reshape(reso, reso),
                     coverages.reshape(reso, reso),
                     reso, cmap="rainbow")
        plt.title("coverage")
        plt.xlabel("gene 1")
        plt.ylabel("gene 2")
        plt.axis("equal")

        plt.subplot(122)
        plt.contourf(genomes[:, 0].reshape(reso, reso),
                     genomes[:, 1].reshape(reso, reso),
                     uniformities.reshape(reso, reso),
                     reso, cmap="rainbow")
        plt.title("uniformity")
        plt.xlabel("gene 1")
        plt.ylabel("gene 2")
        plt.axis("equal")
        plt.show()

        if self.save_figures:
            plt.savefig("figures/{}/evolvability mapping.svg".format(self.experiment_name))
            plt.savefig("figures/{}/evolvability mapping.png".format(self.experiment_name))

    def display_median_fq_tq(self, data_median, data_fq, data_tq, label="", ynorm=False):
        """Displays the median(s), first quartile(s), and third quartile(s) throughout generations.

        Args:
            data_median (np.array(([N_r | 1], N_g), float)): medians
            data_fq (np.array(([N_r | 1], N_g), float)): first quartiles
            data_tq (np.array(([N_r | 1], N_g), float)): third quartiles
            label (str): label to display in the plots
            ynorm (bool): flag to set the y-axis limits to [0, 1]

        """
        assert data_median.ndim == data_fq.ndim == data_tq.ndim, "incorrect data format"
        assert data_median.ndim in [1, 2], "incorrect dimensionality"

        plt.figure(figsize=(9, 6))

        if data_median.ndim == 1:
            generations = np.arange(len(data_median))
            plt.plot(generations, data_median, "b", label=label)
            plt.fill_between(generations, data_fq, data_tq, color="b", alpha=0.25)
            plt.legend()
            plt.title(label)
            plt.xlabel("generations")

        elif data_median.ndim == 2:
            n_runs = data_median.shape[0]
            generations = np.arange(data_median.shape[1])
            for run in range(n_runs):
                color = plt.get_cmap("Dark2")(run % 8)
                plt.plot(generations, data_median[run, :], color=color)
                plt.fill_between(generations, data_fq[run, :], data_tq[run, :], color=color, alpha=0.5)
            plt.title(label)
            plt.xlabel("generations")

        if ynorm:
            plt.ylim([0, 1])
        plt.show()
        if self.save_figures:
            plt.savefig("figures/{}/{}.svg".format(self.experiment_name, label))
            plt.savefig("figures/{}/{}.png".format(self.experiment_name, label))

    def display_all_stats(self):
        """Displays all stats throughout generations.

        """
        self.display_median_fq_tq(self.all_stats["median_ind_coverages"], self.all_stats["fq_ind_coverages"],
                                  self.all_stats["tq_ind_coverages"], label="individual coverage per run", ynorm=True)

        self.display_median_fq_tq(self.all_stats["median_median_ind_coverages"], self.all_stats["fq_median_ind_coverages"],
                                  self.all_stats["tq_median_ind_coverages"], label="individual coverage across runs", ynorm=True)

        self.display_median_fq_tq(self.all_stats["median_ind_uniformities"], self.all_stats["fq_ind_uniformities"],
                                  self.all_stats["tq_ind_uniformities"], label="individual uniformity per run", ynorm=True)

        self.display_median_fq_tq(self.all_stats["median_median_ind_uniformities"], self.all_stats["fq_median_ind_uniformities"],
                                  self.all_stats["tq_median_ind_uniformities"], label="individual uniformity across runs", ynorm=True)

        self.display_median_fq_tq(self.all_stats["median_cum_coverages"], self.all_stats["fq_cum_coverages"],
                                  self.all_stats["tq_cum_coverages"], label="cumulative coverage per run", ynorm=True)

        self.display_median_fq_tq(self.all_stats["median_median_cum_coverages"], self.all_stats["fq_median_cum_coverages"],
                                  self.all_stats["tq_median_cum_coverages"], label="cumulative coverage across runs", ynorm=True)

        self.display_median_fq_tq(self.all_stats["median_cum_uniformities"], self.all_stats["fq_cum_uniformities"],
                                  self.all_stats["tq_cum_uniformities"], label="cumulative uniformity per run", ynorm=True)

        self.display_median_fq_tq(self.all_stats["median_median_cum_uniformities"], self.all_stats["fq_median_cum_uniformities"],
                                  self.all_stats["tq_median_cum_uniformities"], label="cumulative uniformity across runs", ynorm=True)

        self.display_median_fq_tq(self.all_stats["median_ages"], self.all_stats["fq_age"],
                                  self.all_stats["tq_age"], label="individual ages per run")

        self.display_median_fq_tq(self.all_stats["median_median_ages"], self.all_stats["fq_median_ages"],
                                  self.all_stats["tq_median_ages"], label="individual ages across runs")

        self.display_median_fq_tq(self.all_stats["median_distances"], self.all_stats["fq_distances"],
                                  self.all_stats["tq_distances"], label="individual distance to parent per run")

        self.display_median_fq_tq(self.all_stats["median_median_distances"], self.all_stats["fq_median_distances"],
                                  self.all_stats["tq_median_distances"], label="individual distance to parent across runs")

    def display_run_stats(self, run=0):
        """Display the stats throughout the generations for an input run.

        Args:
            run (int): run index

        """
        self.display_median_fq_tq(self.all_stats["median_ind_coverages"][run], self.all_stats["fq_ind_coverages"][run],
                                  self.all_stats["tq_ind_coverages"][run], label="individual coverage for run " + str(run), ynorm=True)

        self.display_median_fq_tq(self.all_stats["median_ind_uniformities"][run], self.all_stats["fq_ind_uniformities"][run],
                                  self.all_stats["tq_ind_uniformities"][run], label="individual uniformity for run " + str(run), ynorm=True)

        self.display_median_fq_tq(self.all_stats["median_cum_coverages"][run], self.all_stats["fq_cum_coverages"][run],
                                  self.all_stats["tq_cum_coverages"][run], label="cumulative coverage for run " + str(run), ynorm=True)

        self.display_median_fq_tq(self.all_stats["median_cum_uniformities"][run], self.all_stats["fq_cum_uniformities"][run],
                                  self.all_stats["tq_cum_uniformities"][run], label="cumulative uniformity for run " + str(run), ynorm=True)

        self.display_median_fq_tq(self.all_stats["median_ages"][run], self.all_stats["fq_age"][run],
                                  self.all_stats["tq_age"][run], label="individual ages for run " + str(run))

        self.display_median_fq_tq(self.all_stats["median_distances"][run], self.all_stats["fq_distances"][run],
                                  self.all_stats["tq_distances"][run], label="individual distance to parent for run " + str(run))

    def display_search_history(self, group="population", run=0, reso=50):
        """Displays the search history for an input run.

        Args:
            group (str): selected group of individuals - can be 'population', 'archive' and 'offsprings'
            run (int): run index
            reso (int): resolution

        """
        assert group in ["population", "archive", "offsprings"], "incorrect group - should be 'population', 'archive', or 'offsprings'"
        if group == "population":
            d = self.all_populations
        elif group == "archive":
            d = self.all_archives
        elif group == "offsprings":
            d = self.all_offsprings
        else:
            return

        xx, yy = np.meshgrid(np.linspace(-5, 5, reso), np.linspace(-5, 5, reso))
        bkgd_genomes = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1)))
        bkgd_behaviors = self.dummy_experiment.gene_to_behavior(bkgd_genomes)
        bkgd_coverages, bkgd_uniformities, _, _ = self.dummy_experiment.evaluate_coverage_and_uniformity(bkgd_genomes)

        genomes = d[run, :, :, :][:, :, I_GENOME]
        behaviors = d[run, :, :, I_BEHAVIOR]
        generations = np.tile(np.arange(d.shape[1]).reshape(-1, 1), (1, d.shape[2]))

        plt.figure(figsize=(16, 12))

        # genome history superimposed on top of behavior
        plt.subplot(221)
        plt.contourf(bkgd_genomes[:, 0].reshape(reso, reso),
                     bkgd_genomes[:, 1].reshape(reso, reso),
                     bkgd_behaviors.reshape(reso, reso),
                     reso, cmap="gray")
        plt.scatter(genomes[:, :, 0].reshape(-1), genomes[:, :, 1].reshape(-1), s=15, c=generations.reshape(-1), cmap="jet")
        plt.title("genomes (on top of behavior) for run " + str(run))
        plt.xlabel("gene 1")
        plt.ylabel("gene 2")
        plt.axis("equal")

        # genome history superimposed on top of coverage
        plt.subplot(222)
        plt.contourf(bkgd_genomes[:, 0].reshape(reso, reso),
                     bkgd_genomes[:, 1].reshape(reso, reso),
                     bkgd_coverages.reshape(reso, reso),
                     reso, cmap="gray")
        plt.scatter(genomes[:, :, 0].reshape(-1), genomes[:, :, 1].reshape(-1), s=15, c=generations.reshape(-1), cmap="jet")
        plt.title("genomes (on top of coverage) for run " + str(run))
        plt.xlabel("gene 1")
        plt.ylabel("gene 2")
        plt.axis("equal")

        # genome history superimposed on top of uniformity
        plt.subplot(223)
        plt.contourf(bkgd_genomes[:, 0].reshape(reso, reso),
                     bkgd_genomes[:, 1].reshape(reso, reso),
                     bkgd_uniformities.reshape(reso, reso),
                     reso, cmap="gray")
        plt.scatter(genomes[:, :, 0].reshape(-1), genomes[:, :, 1].reshape(-1), 15, generations.reshape(-1), cmap="jet")
        plt.title("genomes (on top of uniformity) for run " + str(run))
        plt.xlabel("gene 1")
        plt.ylabel("gene 2")
        plt.axis("equal")

        # behavior history superimposed on top of behavior
        plt.subplot(224)
        plt.scatter(behaviors.reshape(-1), generations.reshape(-1), s=15, c=generations.reshape(-1), cmap="jet")
        plt.title("behavior (on top of behavior) for run " + str(run))
        plt.xlabel("behavior")
        plt.ylabel("generation")

        plt.show()
        if self.save_figures:
            plt.savefig("figures/{}/search_history_run{}.svg".format(self.experiment_name, run))
            plt.savefig("figures/{}/search_history_run{}.png".format(self.experiment_name, run))

    def display_selected_individuals(self, run=0):
        """Displays the individuals selected to be part of the archive and next population at each generation.

        Args:
            run (int): run index

        Returns:

        """

        arch_generations = np.tile(np.arange(self.all_archives.shape[1]).reshape(-1, 1), (1, self.all_archives.shape[2]))
        pop_generations = np.tile(np.arange(self.all_populations.shape[1]).reshape(-1, 1), (1, self.all_populations.shape[2]))
        off_generations = np.tile(np.arange(self.all_offsprings.shape[1]).reshape(-1, 1), (1, self.all_offsprings.shape[2]))

        plt.figure()
        plt.scatter(self.all_archives[run, :, :, I_BEHAVIOR].reshape(-1), arch_generations.reshape(-1), s=20, c="k", marker="o", zorder=3)
        plt.scatter(self.all_populations[run, :, :, I_BEHAVIOR].reshape(-1), pop_generations.reshape(-1) + 0.1, s=20, c="b", marker="o", zorder=4)
        plt.scatter(self.all_offsprings[run, :, :, I_BEHAVIOR].reshape(-1), off_generations.reshape(-1) + 0.2, s=20, c="r", marker="o", zorder=3)

        # background
        plt.hlines(np.arange(self.all_populations.shape[1]) + 0.8, xmin=-6, xmax=6, color=[0.9, 0.9, 0.9], zorder=1)

        # show the offsprings selected for next population
        selected = (self.all_offsprings[run, :, :, I_SELECTED_POP] == 1.)
        behaviors = self.all_offsprings[run, :, :, I_BEHAVIOR][selected]
        generations = off_generations[selected]
        plt.plot(np.tile(behaviors, (2, 1)), np.vstack((generations + 0.2, generations + 1 + 0.1)),
                 color="r", linewidth=2, zorder=2)

        # show the population selected for next population
        selected = (self.all_populations[run, :, :, I_SELECTED_POP] == 1.)
        behaviors = self.all_populations[run, :, :, I_BEHAVIOR][selected]
        generations = pop_generations[selected]
        plt.plot(np.tile(behaviors, (2, 1)), np.vstack((generations + 0.1, generations + 1 + 0.1)),
                 color="b", linewidth=2, zorder=2)

        # show the offsprings selected for archive
        selected = (self.all_offsprings[run, :, :, I_SELECTED_ARC] == 1.)
        behaviors = self.all_offsprings[run, :, :, I_BEHAVIOR][selected]
        generations = off_generations[selected]
        plt.plot(np.tile(behaviors, (2, 1)), np.vstack((generations + 0.2, generations + 1)),
                 color="k", linewidth=2, linestyle="dotted", zorder=2)

        plt.title("selected behaviors for run " + str(run))
        plt.xlabel("behavior")
        plt.ylabel("generation")

        plt.show()
        if self.save_figures:
            plt.savefig("figures/{}/selection_history_run{}.svg".format(self.experiment_name, run))
            plt.savefig("figures/{}/selection_history_run{}.png".format(self.experiment_name, run))

    def display_novelty(self, run=0, generation=0, reso=50):
        """Displays the search history for an input run.

        Args:
            run (int): run index
            generation (int): generation
            reso (int): resolution

        """

        xx, yy = np.meshgrid(np.linspace(-5, 5, reso), np.linspace(-5, 5, reso))
        novelty_genomes = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1)))
        novelty_behaviors = self.dummy_experiment.gene_to_behavior(novelty_genomes)
        old_behaviors = np.hstack((self.all_populations[run, min(generation, self.parameters["frozen"]), :, I_BEHAVIOR],
                                   self.all_offsprings[run, min(generation, self.parameters["frozen"]), :, I_BEHAVIOR],
                                   self.all_archives[run, :(min(generation, self.parameters["frozen"]) + 1), :, I_BEHAVIOR].reshape(-1)))
        novelties = self.dummy_experiment.compute_novelty(novelty_behaviors, old_b=old_behaviors)

        plt.figure(figsize=(12, 6))

        # genomes superimposed on top of novelty
        plt.subplot(121)
        plt.contourf(novelty_genomes[:, 0].reshape(reso, reso),
                     novelty_genomes[:, 1].reshape(reso, reso),
                     novelties.reshape(reso, reso),
                     reso, cmap="rainbow")

        # plot the archive, population, and offsprings
        plt.scatter(self.all_archives[run, :generation + 1, :, :][:, :, I_GENOME[0]].reshape(-1),
                    self.all_archives[run, :generation + 1, :, :][:, :, I_GENOME[1]].reshape(-1), s=15, c="k")
        plt.scatter(self.all_populations[run, generation, :, :][:, I_GENOME[0]].reshape(-1),
                    self.all_populations[run, generation, :, :][:, I_GENOME[1]].reshape(-1), s=15, c="b")
        plt.scatter(self.all_offsprings[run, generation, :, :][:, I_GENOME[0]].reshape(-1),
                    self.all_offsprings[run, generation, :, :][:, I_GENOME[1]].reshape(-1), s=15, c="r")
        plt.title("novelty at generation {} for run {}".format(generation, run))
        plt.xlabel("gene 1")
        plt.ylabel("gene 2")
        plt.axis("equal")

        # behaviors superimposed on top of novelty
        plt.subplot(122)
        order = np.argsort(novelty_behaviors)
        plt.plot(novelty_behaviors[order], novelties[order], color="g")

        # plot the archive, population, and offsprings
        plt.scatter(self.all_archives[run, :generation + 1, :, I_BEHAVIOR].reshape(-1),
                    np.zeros(self.n_selected * (generation + 1)), s=15, c="k")
        plt.scatter(self.all_populations[run, generation, :, I_BEHAVIOR],
                    np.zeros(self.n_pop), s=15, c="b")
        plt.scatter(self.all_offsprings[run, generation, :, I_BEHAVIOR],
                    np.zeros(self.n_offspring), s=15, c="r")
        plt.title("novelty at generation {} for run {}".format(generation, run))
        plt.xlabel("behavior")

        plt.show()
        if self.save_figures:
            plt.savefig("figures/{}/novelty_generation{}_run{}.svg".format(self.experiment_name, generation, run))
            plt.savefig("figures/{}/novelty_generation{}_run{}.png".format(self.experiment_name, generation, run))


if __name__ == "__main__":
    # TODO: remove in the future
    os.chdir("..")

    my_analyzer = Analyzer(save_figures=True)

    my_analyzer.display_mapping()
    my_analyzer.display_mapping_evolvability()
    my_analyzer.display_all_stats()
    my_analyzer.display_run_stats()
    my_analyzer.display_search_history()
    my_analyzer.display_selected_individuals()
    my_analyzer.display_novelty(generation=0)

    plt.show(block=True)

