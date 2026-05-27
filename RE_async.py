"""Parallel body-brain co-evolution with a CLI interface.

Mirrors RE_sync.py but exposes all hyper-parameters as command-line arguments
via Typer and tracks evaluation progress with a Rich progress bar.
"""

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
from robot_worker import LocomotionConfig, run_pool

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
NUM_WORKERS: int = 8

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
        data_dir: Path = Path(CWD / "__data__/ariel_evolution"),
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


class Evo:
    def __init__(
        self,
        config: EvolutionConfig,
        ea_settings: EASettings,
        progress: Progress,
    ) -> None:
        self.config = config
        self.ea_settings = ea_settings
        self.progress = progress
        self.current_gen = 0

        self.spawn_position: tuple[float, float, float] = (0.0, 0.0, 0.1)
        self.target_position: tuple[float, float, float] = (2.0, 0.0, 0.1)

        self.nde = NeuralDevelopmentalEncoding(
            number_of_modules=config.num_modules,
            genotype_size=config.gene_size,
        )
        self.hpd = HighProbabilityDecoder(num_modules=config.num_modules)
        torch.save(self.nde.state_dict(), config.data_dir / "NDE.pth")

    def create_individual(self) -> Individual:
        ind = Individual()
        ind.genotype = self.config.rng.normal(
            loc=0,
            scale=64,
            size=(3, self.config.gene_size),
        ).tolist()
        return ind

    def gene_to_graph(self, genotype: list[list[float]]) -> Any:
        genotype_tensor = torch.tensor(genotype, dtype=torch.float32)
        p_matrices = self.nde.forward(genotype_tensor)
        return self.hpd.probability_matrices_to_graph(
            p_matrices[0],
            p_matrices[1],
            p_matrices[2],
        )

    # ------------------------------------------------------------------ #
    #                         EA Operators                               #
    # ------------------------------------------------------------------ #

    def parent_selection(self, population: Population) -> Population:
        tournament_size = 3
        for ind in population:
            ind.tags["ps"] = False

        num_parents = (len(population) // 2) + 1
        for _ in range(num_parents):
            competitors = [
                list(population)[self.config.rng.integers(len(population))]
                for _ in range(tournament_size)
            ]
            winner = min(competitors, key=lambda ind: ind.fitness)
            winner.tags["ps"] = True
        return population

    def crossover(self, population: Population) -> Population:
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

    def mutation(self, population: Population) -> Population:
        mutation_strength = 0.1
        all_genes = np.array([ind.genotype for ind in population])
        pop_std = float(np.std(all_genes))

        for ind in population:
            if ind.tags.get("mut", False):
                genes = np.array(ind.genotype).flatten()
                noise = self.config.rng.normal(
                    loc=0.0,
                    scale=mutation_strength * pop_std,
                    size=genes.shape,
                )
                ind.genotype = (genes + noise).reshape((3, -1)).tolist()
                ind.tags["mut"] = False
        return population

    def survivor_selection(self, population: Population) -> Population:
        tournament_size = 5

        for ind in population:
            if not hasattr(ind, "alive") or ind.alive is None:
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
            max(candidates, key=lambda ind: ind.fitness).alive = False

        return population

    def evaluate_pop(self, population: Population) -> Population:
        for_eval = [ind for ind in population if ind.requires_eval]
        if not for_eval:
            self.current_gen += 1
            return population

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

            worker_seed = (self.config.seed or 0) + self.current_gen * 10000 + i
            eval_config = LocomotionConfig(
                cma_generations=self.config.cma_generations,
                cma_pop_size=self.config.cma_pop_size,
                spawn_position=self.spawn_position,
                target_position=self.target_position,
                seed=worker_seed,
            )
            eval_args.append((xml_string, eval_config))

        eval_task = self.progress.add_task(
            f"[green]Gen {self.current_gen}: evaluating {len(for_eval)} robots ({NUM_WORKERS} parallel)",
            total=len(for_eval),
        )

        eval_start = time.time()
        results = run_pool(eval_args, num_workers=NUM_WORKERS)
        eval_time = time.time() - eval_start

        self.progress.update(eval_task, completed=len(for_eval))
        self.progress.remove_task(eval_task)

        for ind, fitness in zip(for_eval, results, strict=True):
            ind.fitness = fitness
            ind.requires_eval = False

        console.rule(
            f"Generation {self.current_gen}/{self.ea_settings.num_steps}"
        )
        console.print(
            f"Best: [green]{min(results):.3f}[/] | "
            f"Mean: [blue]{float(np.mean(results)):.3f}[/] | "
            f"Worst: [red]{max(results):.3f}[/] | "
            f"N={len(results)} | took {eval_time:.1f}s",
        )
        self.current_gen += 1

        return population


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
    """Run evolutionary robotics optimisation with parallel evaluation."""
    data_dir = Path.cwd() / output_dir / "ariel_evolution"
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

    console.rule("[bold blue]Ariel Evolutionary Robotics Framework")
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
        evo = Evo(config=config, ea_settings=ea_settings, progress=progress)

        init_task = progress.add_task(
            "[green]Initializing population...", total=pop_size
        )
        population = Population([])
        for _ in range(pop_size):
            population.append(evo.create_individual())
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

        ea = EA(
            population,
            operations=ops,
            num_steps=generations,
        )

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
    ] = Path("__data__/ariel_evolution"),
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
