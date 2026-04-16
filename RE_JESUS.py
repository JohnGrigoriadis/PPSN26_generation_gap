"""
John, synchronous body-brain evolution with J.E.S.U.S.(Joint Evolution Strategies with Undead Sampling).
"""

# Imports
import random
import time
from pathlib import Path
import mujoco as mj
# from mujoco import viewer
import sqlite3
import json
import numpy as np
import pandas as pd
import torch

# Ray for parallelisation
import ray

# Type Checking
from networkx import DiGraph
from typing import Any

# Ariel Imports 

# Learning EA
import nevergrad as ng

# Fix After refactoring
from ariel.ec.a001 import Individual
from ariel.ec.a004 import EA, EASettings, EAStep, Population
from ariel.ec.a005 import Crossover

# Body imports
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller, Tracker
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
        HighProbabilityDecoder,
)

# Import World
from ariel.simulation.environments import (  
        # Simple Flat used in initial testing
        SimpleFlatWorld,
)

# from ariel.utils.runners import simple_runner
# Hivemind related import
from network import Network
from nn_utils import fill_parameters, get_robot_state

# Pretty little errors and progress bars
from rich.console import Console
from rich.traceback import install

# Initialize rich console and traceback handler
install(width=180)
console = Console()
print = console.log

POP_SIZE = 50
NUM_GENERATIONS = 100

RNG = np.random.default_rng()
NUM_MODULES = 20
GENE_SIZE = 64 

CWD = Path.cwd()
SCRIPT_NAME = __file__.split("/")[-1][:-3]
DATA = Path(CWD / "__data__" / SCRIPT_NAME)
DATA.mkdir(exist_ok=True)

# Show cwd
print(f"Current working directory: {CWD}")
print(f"Saving data to {DATA}")

# Lets get this going YYYEEEAAAHHH
config = EASettings(output_folder=DATA,
                    is_maximisation=False,
                    num_of_generations=NUM_GENERATIONS, 
                    db_handling="delete",
                    target_population_size=POP_SIZE 
                    )

# ============================================================================ #
#                           Population Handling                                #
# ============================================================================ #

# Currently Completed
def create_individual(gene_size) -> Individual:
        """
        Create and initialise BODY individual 
        """

        ind = Individual()
        ind.genotype = np.random.normal(loc=0,
                                        scale=64, 
                                        size=(3 , gene_size)).tolist()
        
        return ind

# ------------------------------------------------------------------------ #
#                             EA Operators                                 #
# ------------------------------------------------------------------------ #
class Evo():

    def __init__(self,
                pop_size:int,

                 ) -> None:
        self.current_gen = 0
        
        self.pop_size = pop_size

        # Spawn positions for the 2 testing environments
        self.spawn_position_flat : tuple[float, float, float] = (0.0, 0.0, 0.1)

        # Target positions for the 2 testing environments
        self.target_position_flat: tuple[float, float, float] = (2, 0.0, 0.1)

        # NDE 
        self.nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_MODULES, # Seems to be a good value
                                                genotype_size=64
                                                )
        torch.save(self.nde.state_dict(), DATA/"NDE.pth")

        self.hpd = HighProbabilityDecoder(num_modules=NUM_MODULES)

        self.fit_history = []

        self.conn = sqlite3.connect(DATA)

        # We gotta find JESUS
        self.query = """
            SELECT *
            FROM individual
            WHERE (TIME_OF_DEATH BETWEEN ? AND ?) AND (TIME_OF_BIRTH BETWEEN ? AND ?)
            ORDER BY RANDOM()
            LIMIT ?
        """

    # Completed
    def gene_to_graph(self, genotype):
        """Create mujoco specification from robot genotype"""

        # nde_gene = np.array(genotype).reshape(())
        p_matrices = self.nde.forward(genotype)

        # Decode the high-probability graph
        robot_graph: DiGraph[Any] = self.hpd.probability_matrices_to_graph(
            p_matrices[0],
            p_matrices[1],
            p_matrices[2],
        )

        # robot_spec = construct_mjspec_from_graph(robot_graph)
        return robot_graph
    
    # Completed
    def parent_selection(self, population: Population) -> Population:
        """Tournament Selection"""
        tournament_size: int = 5

        if self.do_we_need_jesus():
            console.log("Summoning JESUS to prevent stagnation...")
            jesus = self.jesi(
                median_age=50, 
                num_jesi=10, 
                tournament_size=tournament_size
            )
            population.extend(jesus)

        # Ensure all individuals have a tags dict and reset parent-selection tag
        for ind in population:
            ind.tags['ps'] = False

        # Decide how many parents we want (even number)
        num_parents = (len(population) // 2) * 2
        if num_parents == 0 and len(population) >= 2:
            num_parents = 2

        # winners : Population = []
        for _ in range(num_parents):
            # sample competitors with replacement
            competitors = [random.choice(population) for _ in range(tournament_size)]

            # pick best competitor depending on maximisation/minimisation
            if config.is_maximisation:
                winner = max(competitors, key=lambda ind: ind.fitness)
            else:
                winner = min(competitors, key=lambda ind: ind.fitness)

            winner.tags['ps'] = True

        return population

    # Completed
    def crossover(self, population: Population) -> Population:
        """One point crossover"""

        parents = [ind for ind in population if ind.tags.get("ps", False)]
        for idx in range(0, len(parents), 2):
            parent_i = parents[idx]
            parent_j = parents[idx]
            genotype_i, genotype_j = Crossover.one_point(parent_i.genotype, 
                                                        parent_j.genotype)

            # First child
            child_i = Individual()
            child_i.genotype = genotype_i
            child_i.tags = {"mut": True}
            child_i.requires_eval = True

            # Second child
            child_j = Individual()
            child_j.genotype = genotype_j
            child_j.tags = {"mut": True}
            child_j.requires_eval = True

            population.extend([child_i, child_j])
        return population

    # Completed
    def mutation(self, population: Population) -> Population:
        """
        Separate mutations for body and hivemind, due to the possibility of different value ranges begin used.
        """
        for ind in population:
            if ind.tags.get("mut", False):
                genes = np.array(ind.genotype).flatten()
                if random.random() < 0.7:
                    
                    mutated = np.array([np.array(i) + np.random.normal(0, 4) for i in genes])
                else:
                    mutated = genes.copy()

                ind.genotype = mutated.reshape((3,-1)).tolist()
                
        return population

    # Completed
    def survivor_selection(self, population: Population) -> Population:

        tournament_size: int = 5

        # Decide how many parents we want (even number)
        pop_len = len(population)

        for _ in range(pop_len):
            # Sample competitors with replacement
            pop_alive = [ind for ind in population if ind.alive is True]
            death_candidates = [random.choice(pop_alive) for _ in range(tournament_size)]

            # Pick best competitor depending on maximisation/minimisation
            if config.is_maximisation:
                about_to_be_killed_lol = min(death_candidates, key=lambda ind: ind.fitness)
            else:
                about_to_be_killed_lol = max(death_candidates, key=lambda ind: ind.fitness)

            about_to_be_killed_lol.alive = False

            pop_len -= 1
            if pop_len <= self.pop_size:
                break

        return population
    
    def do_we_need_jesus(self, 
                         window: int = 10, 
                         threshold: float = 0.2) -> bool:
        """
        Returns True if the most recent generation's mean fitness is stagnating
        relative to the previous window generations.
        """
        if len(self.fit_history) < window + 1:
            return False

        current = self.fit_history[-1]
        window_mean = np.mean(self.fit_history[-window - 1 : -1])

        return bool(np.isclose(current, window_mean, rtol=threshold))

    #? Work in Progress
    def jesi(self,
             median_age: int,
             num_jesi: int,
             tournament_size: int,
            ) -> list[Individual]:
        
        assert tournament_size <= num_jesi, "Tournament size must be <= JESI individuals."

        time_death_low = max(0, median_age - 10)
        time_death_high = median_age

        time_birth_low = 0
        time_birth_high = time_death_low - 5

        # Fetch a larger pool so tournaments have meaningful competition
        pool_size = num_jesi * tournament_size

        df = pd.read_sql_query(
            self.query,
            self.conn,
            params=(time_death_low, time_death_high, 
                    time_birth_low, time_birth_high, 
                    pool_size
                    ),
        )

        # Parse JSON string columns back into Python objects
        for col in ("genotype_", "tags_"):
            df[col] = df[col].apply(lambda v: json.loads(v) if isinstance(v, str) else v)

        pool = [Individual.model_validate(row.to_dict()) for _, row in df.iterrows()]

        # Run num_jesi tournaments, each of size tournament_size
        winners = []
        for _ in range(num_jesi):
            contestants = [random.choice(pool) for i in range(tournament_size)]
            winners.append(max(contestants, key=lambda ind: ind.fitness))

        return winners
    
    # Completed
    def evaluate_pop(self, population : Population) -> Population:

        # Turn all NDEs into graphs so we don't have to decode 
        # them in the eval function

        for_eval = [ind for ind in population if ind.requires_eval]
        robot_graphs = [self.gene_to_graph(ind.genotype) for ind in for_eval]
        num_inds = len(for_eval)

        eval_start_time = time.time()

        # Init parallel tasks
        task_ids = []
        for robot in robot_graphs:
            oid = evaluate_pair_worker.remote(
                robot,
                self.spawn_position_flat,
                self.target_position_flat,
            )
            task_ids.append(oid)

        # Get all the results
        results = ray.get(task_ids)

        # error was here, i was giving the wrong fitnesses
        # Iterte over pop and fill in the missing fitness values
        idx_pop = 0
        idx_for_eval = 0
        while idx_pop < len(population) and idx_for_eval < len(for_eval):
            ind = population[idx_pop]
            if ind.requires_eval:
                ind.fitness = results[idx_for_eval]
                ind.requires_eval = False
                idx_for_eval += 1
            idx_pop += 1

        eval_end_time = time.time()
        console.rule(f"Generation {self.current_gen}/{config.num_of_generations}")
        print(f"Best Fitness: {np.min(results):.3f}")
        print(f"Mean Fitness: {np.mean(results):.3f}")
        print(f"Number individuals tested: {num_inds}")
        print(f"Gen {self.current_gen} took {eval_end_time-eval_start_time:.3f} seconds")
        self.current_gen += 1

        return population


@ray.remote(num_cpus=6)
def evaluate_pair_worker(
    robot_graph,
    spawn_pos: tuple[float, float, float],
    target_pos: tuple[float, float, float],
    )-> float:
    """
    Remote worker that constructs a body and evaluates it with a hivemind.
    Accepts nde_model explicitly to avoid global variable dependency.
    """

    robot_spec = construct_mjspec_from_graph(robot_graph).spec

    # 2. Simulation Setup (Logic from evaluate_single)
    mj.set_mjcb_control(None)
    world = SimpleFlatWorld()
    world.spawn(
        robot_spec,
        position=spawn_pos,
        correct_collision_with_floor=True,
    )

    model = world.spec.compile()
    data = mj.MjData(model)  # type:ignore

    input_size = len(get_robot_state(data, target_position=target_pos))
    # print(input_size)
    output_size = model.nu  # type:ignore

    if model.nu < 2: # type:ignore
        # return bad fitness if robot kinda cannot move
        # made to be adaptabel to different target positions
        return target_pos[0]

    lr_pop_size = 10 
    generations = 10

    min_fit = np.inf

    net = Network(input_size=input_size, hidden_size=16, output_size=output_size)
    num_vars = sum(p.numel() for p in net.parameters())

    local_learner = ng.optimizers.CMA(
        parametrization = num_vars,
        budget = lr_pop_size * generations,
    )

    tracker = Tracker(name_to_bind="core", observable_attributes=["xpos"], quiet=True)
    tracker.setup(world.spec, data)

    controller = Controller(controller_callback_function=net.forward, tracker=tracker)
    for _ in range(generations):
        vecs = [local_learner.ask() for _ in range(lr_pop_size)]

        for vec_candidate in vecs:
            # 3. Network Construction
            fill_parameters(net, vec_candidate.value)

            mj.mj_resetData(model, data)  # type:ignore
            # Compensate for "flop"
            simple_runner(model, data, duration=3) # type:ignore
            displacement = data.qpos[0:3].copy()

            # Add "flop" displacement to target so needed distance stays the same.
            # Should not be used with olympic arena for now...
            (xt, yt, zt) = target_pos  + displacement

            mj.set_mjcb_control(lambda m, d: controller.set_control(m, d, target_position=(xt, yt, zt)))
            # 4. Run Simulation
            simple_runner(model, data, duration=15)  # type: ignore

            # 5. Calculate Fitness
            xc, yc, zc = data.qpos[0:3].copy()
            fitness = np.sqrt((xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2)

            local_learner.tell(vec_candidate, fitness)

            min_fit = min(min_fit, fitness)

    return min_fit

# ------------------------------------------------------------------------ #
#                           Helper Functions                               #
# ------------------------------------------------------------------------ #

# Currently Completed
def simple_runner(
    model: mj.MjModel,
    data: mj.MjData,
    duration: float = 10.0,
    steps_per_loop: int = 100,
) -> None:
    """
    Run a simple headless simulation for a given duration, 
    *without resetting the simulation.

    Parameters
    ----------
    model : mujoco.MjModel
        The MuJoCo model to simulate.
    data : mujoco.MjData
        The MuJoCo data to simulate.
    duration : float, optional
        The duration of the simulation in seconds, by default 10.0
    steps_per_loop : int, optional
        The number of simulation steps to take in each loop, by default 100
    """

    # Define action specification and set policy
    data.ctrl = RNG.normal(scale=0.1, size=model.nu)  # type: ignore

    while data.time < duration:
        mj.mj_step(model, data, nstep=steps_per_loop)

# ------------------------------------------------------------------------ #
#                        Main Evolution Loop                               #
# ------------------------------------------------------------------------ #

def evolve() -> EA:

    console.log("Initializing population...")
    hivemind_EA = Evo(pop_size=POP_SIZE)
    
    # Initialise Body & Hivemind Population
    population = [create_individual(GENE_SIZE) for _ in range(POP_SIZE)]

    # Initial Eval
    population = hivemind_EA.evaluate_pop(population)

    # Define Evolution Loop
    # Operators work for both NDEs and Network Weight Vectors 
    ops = [    
        # Default EA operators
        EAStep("parent_selection", hivemind_EA.parent_selection),
        EAStep("crossover", hivemind_EA.crossover),
        EAStep("mutation", hivemind_EA.mutation),
        EAStep("evaluation", hivemind_EA.evaluate_pop),
        EAStep("survivor_selection", hivemind_EA.survivor_selection),
    ]

    # Initialise EA object
    ea = EA(population, 
            operations=ops,
            num_of_generations=NUM_GENERATIONS
                )
    
    ea.run()

    return ea

def main():

    # Initialize Ray. ignore_reinit_error=True helps if you run this in a notebook/loop
    ray.init(ignore_reinit_error=True)

    _ = evolve()
    # torch.save(best_hivemind.genotype, Path("__data__/best_hivemind_data.pth"))
    ray.shutdown()
    
if __name__ == "__main__":
   
    start = time.time()

    main()
    
    end = time.time()

    time_taken = end-start

    # Literally just to see the results better while testing
    if time_taken < 60:
        print(f"Code took {time_taken:.3f} seconds to run") 
    elif time_taken < 60*60:
        print(f"Code took {time_taken/60:.3f} minutes to run") 
    else:
        print(f"Code took {time_taken/(60*60):.3f} hours to run") 