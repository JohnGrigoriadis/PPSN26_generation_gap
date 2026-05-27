"""Synchronous body-brain co-evolution baseline."""

import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import torch
import typer
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
)
from ariel.ec import (
    EA,
    Crossover,
    EAOperation,
    EASettings,
    Individual,
    Population,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.environments import SimpleFlatWorld
from networkx import DiGraph
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
from rich.traceback import install
from robot_worker import LocomotionConfig, run_pool

install(width=180)
console = Console()

app = typer.Typer(
    name="re-sync",
    help="Synchronous body-brain co-evolution baseline.",
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)

DEFAULT_POP_SIZE: int = 50
DEFAULT_NUM_GENERATIONS: int = 100
DEFAULT_NUM_MODULES: int = 20
DEFAULT_GENE_SIZE: int = 64
DEFAULT_CMA_GENERATIONS: int = 20
DEFAULT_CMA_POP_SIZE: int = 10
NUM_WORKERS: int = 6

CWD = Path.cwd()


class EvolutionConfig:
    def __init__(
        self,
        pop_size: int = DEFAULT_POP_SIZE,
        num_generations: int = DEFAULT_NUM_GENERATIONS,
        num_modules: int = DEFAULT_NUM_MODULES,
        gene_size: int = DEFAULT_GENE_SIZE,
        cma_generations: int = DEFAULT_CMA_GENERATIONS,
        cma_pop_size: int = DEFAULT_CMA_POP_SIZE,
        data_dir: Path = Path(CWD / "__data__/RE_sync"),
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


def create_individual(gene_size: int) -> Individual:
    ind = Individual()
    ind.genotype = np.random.normal(
        loc=0, scale=64, size=(3, gene_size)
    ).tolist()
    return ind


class Evo:
    def __init__(
        self, config: EvolutionConfig, ea_settings: EASettings
    ) -> None:
        self.current_gen = 0
        self.pop_size = config.pop_size
        self.spawn_position: tuple[float, float, float] = (0.0, 0.0, 0.1)
        self.target_position: tuple[float, float, float] = (2.0, 0.0, 0.1)

        self.nde = NeuralDevelopmentalEncoding(
            number_of_modules=config.num_modules,
            genotype_size=config.gene_size,
        )
        torch.save(self.nde.state_dict(), config.data_dir / "NDE.pth")

        self.hpd = HighProbabilityDecoder(num_modules=config.num_modules)
        self.config = ea_settings
        self.evo_config = config

    def gene_to_graph(self, genotype) -> DiGraph[Any]:
        p_matrices = self.nde.forward(genotype)
        return self.hpd.probability_matrices_to_graph(
            p_matrices[0],
            p_matrices[1],
            p_matrices[2],
        )

    def parent_selection(self, population: Population) -> Population:
        tournament_size = 3
        for ind in population:
            ind.tags["ps"] = False

        num_parents = (len(population) // 2) + 1
        if num_parents == 0 and len(population) >= 2:
            num_parents = 2

        for _ in range(num_parents):
            competitors = [
                random.choice(list(population)) for _ in range(tournament_size)
            ]
            if self.config.is_maximisation:
                winner = max(competitors, key=lambda ind: ind.fitness)
            else:
                winner = min(competitors, key=lambda ind: ind.fitness)
            winner.tags["ps"] = True

        return population

    def crossover(self, population: Population) -> Population:
        parents = [ind for ind in population if ind.tags.get("ps", False)]

        num_children = 0
        while num_children < self.config.target_population_size // 2:
            idx_i, idx_j = np.random.choice(len(parents), size=2, replace=False)
            parent_i: Individual = parents[idx_i]
            parent_j: Individual = parents[idx_j]
            genotype_i, genotype_j = Crossover.one_point(
                parent_i.genotype, parent_j.genotype
            )

            child_i = Individual()
            child_i.genotype = genotype_i
            child_i.tags = {"mut": True}
            child_i.requires_eval = True

            child_j = Individual()
            child_j.genotype = genotype_j
            child_j.tags = {"mut": True}
            child_j.requires_eval = True

            population.extend([child_i, child_j])
            num_children += 2

        return population

    def mutation(self, population: Population) -> Population:
        for ind in population:
            if ind.tags.get("mut", False):
                genes = np.array(ind.genotype).flatten()
                if random.random() < 0.7:
                    mutated = genes + np.random.normal(0, 4, size=genes.shape)
                else:
                    mutated = genes.copy()
                ind.genotype = mutated.reshape((3, -1)).tolist()
        return population

    def survivor_selection(self, population: Population) -> Population:
        reverse = self.config.is_maximisation
        ranked = sorted(
            population, key=lambda ind: ind.fitness, reverse=reverse
        )

        assert (
            ranked[0].fitness <= ranked[-1].fitness
            or self.config.is_maximisation
        ), "Ranking error: best fitness is not ranked first"

        for ind in ranked[self.pop_size :]:
            ind.alive = False

        return population

    def evaluate_pop(self, population: Population) -> Population:
        for_eval = [ind for ind in population if ind.requires_eval]
        non_eval = [ind for ind in population if not ind.requires_eval]

        eval_args = []
        for i, ind in enumerate(for_eval):
            robot_graph = self.gene_to_graph(ind.genotype)
            robot_spec_obj = construct_mjspec_from_graph(robot_graph)

            world = SimpleFlatWorld()
            world.spawn(
                robot_spec_obj.spec,
                position=self.spawn_position,
                correct_collision_with_floor=True,
            )
            xml_string = world.spec.to_xml()

            eval_config = LocomotionConfig(
                cma_generations=self.evo_config.cma_generations,
                cma_pop_size=self.evo_config.cma_pop_size,
                spawn_position=self.spawn_position,
                target_position=self.target_position,
                seed=self.current_gen * 10000 + i,
            )
            eval_args.append((xml_string, eval_config))

        eval_start = time.time()
        results = run_pool(eval_args, num_workers=NUM_WORKERS)
        eval_time = time.time() - eval_start

        for ind, res in zip(for_eval, results, strict=True):
            ind.fitness = res
            ind.requires_eval = False

        console.rule(f"Generation {self.current_gen}/{self.config.num_steps}")
        console.log(f"Best:  {np.min(results):.3f}")
        console.log(f"Mean:  {np.mean(results):.3f}")
        console.log(f"N={len(results)} | took {eval_time:.1f}s")
        self.current_gen += 1

        return Population(non_eval + for_eval)


def _print_summary(
    population: Population,
    total_time: float,
    db_name: str,
    gens: int,
) -> None:
    console.rule("[bold green]Evolution Complete")
    table = Table(title="Final Statistics", show_header=True)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    fitnesses = [
        ind.fitness_
        for ind in population
        if hasattr(ind, "fitness_") and ind.fitness_ is not None
    ]
    if fitnesses:
        table.add_row("Best Fitness", f"{min(fitnesses):.6f}")
        table.add_row("Mean Fitness", f"{float(np.mean(fitnesses)):.6f}")
        table.add_row("Worst Fitness", f"{max(fitnesses):.6f}")
        table.add_row("Std Fitness", f"{float(np.std(fitnesses)):.6f}")

    table.add_row("Total Time", str(timedelta(seconds=int(total_time))))
    if gens > 0:
        table.add_row("Time per Generation", f"{total_time / gens:.1f}s")
    table.add_row("Database", db_name)
    console.print(table)


@app.command()
def evolve(
    pop_size: Annotated[
        int,
        typer.Option("--pop-size", "-p", help="Population size", min=2),
    ] = DEFAULT_POP_SIZE,
    generations: Annotated[
        int,
        typer.Option("--generations", "-g", help="Generations", min=1),
    ] = DEFAULT_NUM_GENERATIONS,
    num_modules: Annotated[
        int,
        typer.Option("--modules", "-m", help="Max body modules", min=1),
    ] = DEFAULT_NUM_MODULES,
    gene_size: Annotated[
        int,
        typer.Option("--gene-size", help="Genotype size per layer", min=1),
    ] = DEFAULT_GENE_SIZE,
    cma_generations: Annotated[
        int,
        typer.Option("--cma-gen", help="CMA-ES generations per eval", min=1),
    ] = DEFAULT_CMA_GENERATIONS,
    cma_pop_size: Annotated[
        int,
        typer.Option("--cma-pop", help="CMA-ES population", min=2),
    ] = DEFAULT_CMA_POP_SIZE,
    seed: Annotated[
        int | None,
        typer.Option("--seed", "-s", help="Random seed"),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory"),
    ] = Path("__data__"),
) -> None:
    """Run synchronous body-brain co-evolution."""
    data_dir = Path.cwd() / output_dir / "RE_sync"
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

    db_name = f"database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
    ea_settings = EASettings(
        output_folder=data_dir,
        db_file_name=db_name,
        is_maximisation=False,
        num_steps=generations,
        db_handling="halt",
        target_population_size=pop_size,
    )

    console.rule("[bold blue]RE_sync — Synchronous Body-Brain Evolution")
    console.print(f"Data directory: [green]{data_dir}[/]")
    console.print(f"Population size: [yellow]{pop_size}[/]")
    console.print(f"Generations: [yellow]{generations}[/]")
    console.print(f"Parallel workers: [yellow]{NUM_WORKERS}[/]")
    console.print(
        f"CMA-ES config: [yellow]{cma_generations} gen × {cma_pop_size} pop[/]"
    )
    console.print(f"Random seed: [yellow]{seed or 'Random'}[/]")
    console.print(f"Database: [green]{db_name}[/]")
    console.print()

    start_time = time.perf_counter()

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
        evo = Evo(config=config, ea_settings=ea_settings)

        init_task = progress.add_task(
            "[green]Initializing population...", total=pop_size
        )
        population = Population([])
        for _ in range(pop_size):
            population.append(create_individual(gene_size))
            progress.advance(init_task)
        progress.remove_task(init_task)

        population = evo.evaluate_pop(population)

        ops = [
            EAOperation(evo.parent_selection),
            EAOperation(evo.crossover),
            EAOperation(evo.mutation),
            EAOperation(evo.evaluate_pop),
            EAOperation(evo.survivor_selection),
        ]

        ea = EA(population, operations=ops, num_steps=generations)

        try:
            ea.run()
        except KeyboardInterrupt:
            console.print("\n[yellow]Evolution interrupted by user[/]")
            raise typer.Exit(code=1)

        final_population = ea.population

    total_time = time.perf_counter() - start_time
    _print_summary(final_population, total_time, db_name, evo.current_gen)


@app.command()
def status(
    data_dir: Annotated[
        Path,
        typer.Argument(help="Path to __data__ directory"),
    ] = Path("__data__/RE_sync"),
) -> None:
    """Check status of existing evolution runs."""
    data_dir = Path.cwd() / data_dir
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
