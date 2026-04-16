"""Evolutionary Robotics runner with parallel evaluation.

Uses multiprocessing to run 8 parallel simulations (batch size 8).
"""

import random
import time
from datetime import timedelta
from multiprocessing import get_context
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import torch
import typer
from mujoco_worker import EvalConfig, evaluate_individual
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# Local project imports
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
)
from ariel.ec.a001 import Individual
from ariel.ec.a004 import EASettings, Population
from ariel.ec.a005 import Crossover
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.environments import SimpleFlatWorld

console = Console()

app = typer.Typer(
    name="ariel-evolve",
    help="Evolutionary Robotics Framework using Neural Developmental Encoding",
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)

DEFAULT_POP_SIZE: int = 50
DEFAULT_NUM_GENERATIONS: int = 100
DEFAULT_NUM_MODULES: int = 20
DEFAULT_GENE_SIZE: int = 64
DEFAULT_CMA_GENERATIONS: int = 20
DEFAULT_CMA_POP_SIZE: int = 5


class EvolutionConfig:
    def __init__(
        self,
        pop_size: int = DEFAULT_POP_SIZE,
        num_generations: int = DEFAULT_NUM_GENERATIONS,
        num_modules: int = DEFAULT_NUM_MODULES,
        gene_size: int = DEFAULT_GENE_SIZE,
        cma_generations: int = DEFAULT_CMA_GENERATIONS,
        cma_pop_size: int = DEFAULT_CMA_POP_SIZE,
        data_dir: Path = Path("__data__/ariel_evolution"),
        seed: int | None = None,
    ) -> None:
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.num_modules = num_modules
        self.gene_size = gene_size
        self.cma_generations = cma_generations
        self.cma_pop_size = cma_pop_size
        self.data_dir = data_dir
        self.seed = seed
        self.rng = np.random.default_rng(seed)


class EvolutionOrchestrator:
    def __init__(
        self,
        config: EvolutionConfig,
        console: Console,
        progress: Progress,
    ) -> None:
        self.config = config
        self.console = console
        self.progress = progress
        self.current_gen = 0

        self.spawn_position: tuple[float, float, float] = (0.0, 0.0, 0.1)
        self.target_position: tuple[float, float, float] = (2.0, 0.0, 0.1)

        # NDE setup
        self.nde = NeuralDevelopmentalEncoding(
            number_of_modules=config.num_modules,
            genotype_size=config.gene_size,
        )
        self.hpd = HighProbabilityDecoder(num_modules=config.num_modules)

        torch.save(self.nde.state_dict(), config.data_dir / "NDE.pth")

        # EA Settings
        db_name = self._generate_unique_db_name(config.data_dir)
        self.ea_settings = EASettings(
            output_folder=config.data_dir,
            db_file_name=db_name,
            is_maximisation=False,
            num_of_generations=config.num_generations,
            db_handling="halt",
            target_population_size=config.pop_size,
        )

        self._gen_task_id: int | None = None

    def _generate_unique_db_name(
        self, data_dir: Path, base_name: str = "database"
    ) -> str:
        counter = 0
        db_name = f"{base_name}.db"
        while (data_dir / db_name).exists():
            counter += 1
            db_name = f"{base_name}_{counter}.db"
        return db_name

    def create_individual(self) -> Individual:
        ind = Individual()
        ind.genotype = self.config.rng.normal(
            loc=0, scale=64, size=(3, self.config.gene_size)
        ).tolist()
        return ind

    def gene_to_graph(self, genotype: list[list[float]]) -> Any:
        genotype_tensor = torch.tensor(genotype, dtype=torch.float32)
        p_matrices = self.nde.forward(genotype_tensor)
        robot_graph = self.hpd.probability_matrices_to_graph(
            p_matrices[0], p_matrices[1], p_matrices[2]
        )
        return robot_graph

    def run(self) -> None:
        start_time = time.perf_counter()

        self._gen_task_id = self.progress.add_task(
            "[cyan]Initializing Evolution...",
            total=self.config.num_generations,
        )

        self.console.rule("[bold blue]Ariel Evolutionary Robotics Framework")
        self.console.print(f"Data directory: [green]{self.config.data_dir}[/]")
        self.console.print(
            f"Population size: [yellow]{self.config.pop_size}[/]"
        )
        self.console.print(
            f"Generations: [yellow]{self.config.num_generations}[/]"
        )
        self.console.print(f"Parallel workers: [yellow]8[/] (batch size 8)")
        self.console.print(
            f"CMA-ES config: [yellow]{self.config.cma_generations} gen × {self.config.cma_pop_size} pop[/]"
        )
        self.console.print(
            f"Random seed: [yellow]{self.config.seed or 'Random'}[/]"
        )
        self.console.print()

        # Init population
        init_task = self.progress.add_task(
            "[green]Initializing population...", total=self.config.pop_size
        )
        population: list[Individual] = []
        for _ in range(self.config.pop_size):
            population.append(self.create_individual())
            self.progress.advance(init_task)
        self.progress.remove_task(init_task)

        # Evaluate initial population
        population = self.evaluate_population(population)

        # Main loop
        for gen in range(self.config.num_generations):
            self.current_gen = gen + 1

            self.progress.update(
                self._gen_task_id,
                description=f"[cyan]Generation {self.current_gen}/{self.config.num_generations}",
            )

            population = self.parent_selection(population)
            population = self.crossover(population)
            population = self.mutation(population)
            population = self.evaluate_population(population)
            population = self.survivor_selection(population)

            # Log stats
            fitnesses = [
                ind.fitness
                for ind in population
                if hasattr(ind, "fitness") and ind.fitness is not None
            ]
            if fitnesses:
                self.console.print(
                    f"[Gen {self.current_gen:3d}] "
                    f"Best: [green]{min(fitnesses):.4f}[/] | "
                    f"Mean: [blue]{float(np.mean(fitnesses)):.4f}[/] | "
                    f"Worst: [red]{max(fitnesses):.4f}[/]"
                )

            self.progress.advance(self._gen_task_id)

        total_time = time.perf_counter() - start_time
        self._print_summary(population, total_time)

    def evaluate_population(
        self, population: list[Individual]
    ) -> list[Individual]:
        """Evaluate population using 8 parallel workers."""
        for_eval = [ind for ind in population if ind.requires_eval]
        if not for_eval:
            return population

        # Prepare (xml_string, config) tuples for workers
        eval_args = []
        for i, ind in enumerate(for_eval):
            # Convert genotype to XML
            robot_graph = self.gene_to_graph(ind.genotype)
            robot_spec_obj = construct_mjspec_from_graph(robot_graph)

            # Build world and embed robot
            world = SimpleFlatWorld()
            world.spawn(
                robot_spec_obj.spec,
                position=self.spawn_position,
                correct_collision_with_floor=True,
            )
            xml_string = world.spec.to_xml()

            # Unique seed for reproducibility in worker
            worker_seed = (self.config.seed or 0) + self.current_gen * 10000 + i

            eval_config = EvalConfig(
                cma_generations=self.config.cma_generations,
                cma_pop_size=self.config.cma_pop_size,
                spawn_position=self.spawn_position,
                target_position=self.target_position,
                seed=worker_seed,
            )
            eval_args.append((xml_string, eval_config))

        # Parallel evaluation with progress bar
        eval_task = self.progress.add_task(
            f"[green]Evaluating {len(for_eval)} robots (8 parallel)",
            total=len(for_eval),
        )

        # Use spawn context to avoid MuJoCo/OpenMP thread issues
        ctx = get_context("spawn")

        # Run 8 parallel processes
        with ctx.Pool(processes=8) as pool:
            results = []
            # imap preserves order so results[i] matches for_eval[i]
            for fitness in pool.imap(
                evaluate_individual, eval_args, chunksize=1
            ):
                results.append(fitness)
                self.progress.advance(eval_task)

        self.progress.remove_task(eval_task)

        # Assign fitness back
        for ind, fitness in zip(for_eval, results):
            ind.fitness = fitness
            ind.requires_eval = False

        return population

    def parent_selection(
        self, population: list[Individual]
    ) -> list[Individual]:
        tournament_size = 3
        for ind in population:
            ind.tags["ps"] = False

        num_parents = (len(population) // 2) + 1

        for _ in range(num_parents):
            competitors = [
                population[self.config.rng.integers(len(population))]
                for _ in range(tournament_size)
            ]
            winner = min(competitors, key=lambda ind: ind.fitness)
            winner.tags["ps"] = True
        return population

    def crossover(self, population: list[Individual]) -> list[Individual]:
        parents = [ind for ind in population if ind.tags.get("ps", False)]
        num_children = self.ea_settings.target_population_size // 2

        for _ in range(num_children):
            if len(parents) < 2:
                break
            idx_i, idx_j = self.config.rng.choice(
                len(parents), size=2, replace=False
            )
            p1, p2 = parents[idx_i], parents[idx_j]
            g1, g2 = Crossover.one_point(p1.genotype, p2.genotype)

            c1 = Individual()
            c1.genotype = g1
            c1.tags = {"mut": True}
            c1.requires_eval = True
            population.append(c1)

            if len(population) < self.ea_settings.target_population_size:
                c2 = Individual()
                c2.genotype = g2
                c2.tags = {"mut": True}
                c2.requires_eval = True
                population.append(c2)
        return population

    def mutation(self, population: list[Individual]) -> list[Individual]:
        mutation_strength = 0.1
        all_genes = np.array([ind.genotype for ind in population])
        pop_std = float(np.std(all_genes))

        for ind in population:
            if ind.tags.get("mut", False):
                genes = np.array(ind.genotype).flatten()
                noise = self.config.rng.normal(
                    loc=0.0, scale=mutation_strength * pop_std, size=genes.shape
                )
                mutated = genes + noise
                ind.genotype = mutated.reshape((3, -1)).tolist()
                ind.tags["mut"] = False
        return population

    def survivor_selection(
        self, population: list[Individual]
    ) -> list[Individual]:
        tournament_size = 5

        for ind in population:
            if not hasattr(ind, "alive"):
                ind.alive = True

        while (
            len([ind for ind in population if ind.alive])
            > self.ea_settings.target_population_size
        ):
            alive = [ind for ind in population if ind.alive]
            if not alive:
                break
            candidates = [
                alive[self.config.rng.integers(len(alive))]
                for _ in range(min(tournament_size, len(alive)))
            ]
            to_kill = max(candidates, key=lambda ind: ind.fitness)
            to_kill.alive = False

        return [ind for ind in population if ind.alive]

    def _print_summary(
        self, population: list[Individual], total_time: float
    ) -> None:
        self.console.rule("[bold green]Evolution Complete")
        table = Table(title="Final Statistics", show_header=True)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")

        fitnesses = [
            ind.fitness
            for ind in population
            if hasattr(ind, "fitness") and ind.fitness is not None
        ]
        if fitnesses:
            table.add_row("Best Fitness", f"{min(fitnesses):.6f}")
            table.add_row("Mean Fitness", f"{float(np.mean(fitnesses)):.6f}")
            table.add_row("Worst Fitness", f"{max(fitnesses):.6f}")
            table.add_row("Std Fitness", f"{float(np.std(fitnesses)):.6f}")

        table.add_row("Total Time", str(timedelta(seconds=int(total_time))))
        table.add_row(
            "Time per Generation", f"{total_time / self.current_gen:.1f}s"
        )
        table.add_row("Database", self.ea_settings.db_file_name)
        self.console.print(table)


@app.command()
def evolve(
    pop_size: Annotated[
        int, typer.Option("--pop-size", "-p", help="Population size", min=2)
    ] = DEFAULT_POP_SIZE,
    generations: Annotated[
        int, typer.Option("--generations", "-g", help="Generations", min=1)
    ] = DEFAULT_NUM_GENERATIONS,
    num_modules: Annotated[
        int, typer.Option("--modules", "-m", help="Max body modules", min=1)
    ] = DEFAULT_NUM_MODULES,
    gene_size: Annotated[
        int, typer.Option("--gene-size", help="Genotype size per layer", min=1)
    ] = DEFAULT_GENE_SIZE,
    cma_generations: Annotated[
        int,
        typer.Option("--cma-gen", help="CMA-ES generations per eval", min=1),
    ] = DEFAULT_CMA_GENERATIONS,
    cma_pop_size: Annotated[
        int, typer.Option("--cma-pop", help="CMA-ES population", min=2)
    ] = DEFAULT_CMA_POP_SIZE,
    seed: Annotated[
        int | None, typer.Option("--seed", "-s", help="Random seed")
    ] = None,
    output_dir: Annotated[
        Path, typer.Option("--output", "-o", help="Output directory")
    ] = Path("__data__"),
) -> None:
    """Run evolutionary robotics optimization with parallel evaluation."""
    script_name = "ariel_evolution"
    data_dir = output_dir / script_name
    data_dir.mkdir(parents=True, exist_ok=True)

    config = EvolutionConfig(
        pop_size=pop_size,
        num_generations=generations,
        num_modules=num_modules,
        gene_size=gene_size,
        cma_generations=cma_generations,
        cma_pop_size=cma_pop_size,
        data_dir=data_dir,
        seed=seed,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="green", finished_style="green"),
        TaskProgressColumn(),
        TimeRemainingColumn(elapsed_when_finished=True),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        console=console,
        transient=False,
    ) as progress:
        orchestrator = EvolutionOrchestrator(config, console, progress)
        try:
            orchestrator.run()
        except KeyboardInterrupt:
            console.print("\n[yellow]Evolution interrupted by user[/]")
            raise typer.Exit(code=1)


@app.command()
def status(
    data_dir: Annotated[
        Path, typer.Argument(help="Path to __data__ directory")
    ] = Path("__data__/ariel_evolution"),
) -> None:
    """Check status of existing evolution runs."""
    if not data_dir.exists():
        console.print(f"[red]Directory not found:[/] {data_dir}")
        return

    db_files = list(data_dir.glob("*.db"))
    nde_files = list(data_dir.glob("*.pth"))

    table = Table(title=f"Evolution Status: {data_dir}")
    table.add_column("File Type", style="cyan")
    table.add_column("Count", style="green")
    table.add_column("Latest", style="yellow")

    table.add_row(
        "Databases",
        str(len(db_files)),
        max([f.name for f in db_files] or ["N/A"]),
    )
    table.add_row(
        "NDE Checkpoints",
        str(len(nde_files)),
        max([f.name for f in nde_files] or ["N/A"]),
    )
    console.print(table)


if __name__ == "__main__":
    app()
