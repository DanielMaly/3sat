import os
import click
import glob
import time

from three_sat.models import Instance
from .generator import generate_instance
from .algorithms import solve_genetic


@click.group()
def cli():
    """
    Entry point for the application's CLI
    """
    pass


@cli.command()
@click.option('--literals', '-l', type=click.INT, prompt='How many variables should the instances have?',
              help='The number of literals.')
@click.option('--count', '-c', default=1, help='Number of instances to generate.')
@click.option('--ratio', '-r', type=click.FLOAT, default=3.0, help='The ratio of clauses to variables')
@click.option('--weight-min', '-m', type=click.INT, default=1, help='The minimum clause weight')
@click.option('--weight-max', '-x', type=click.INT, default=30, help='The maximum clause weight')
@click.argument('out_directory')
def generate(literals, count, ratio, weight_min, weight_max, out_directory):
    """
    Generates instances of the 3SAT problem.
    """
    if not os.path.isdir(out_directory):
        print('ERROR: {} is not a directory'.format(out_directory))
        return

    instances = [generate_instance(literals, ratio, weight_min, weight_max) for i in range(count)]

    for i in range(count):
        with open(os.path.join(out_directory, '{0:03d}.inst.dat'.format(i)), mode='w') as out:
            out.write(str(instances[i]))


@cli.command()
@click.argument('in_path')
def solve(in_path):
    instances = []

    if os.path.isfile(in_path):
        instances.append(Instance.from_file(in_path))

    elif os.path.isdir(in_path):
        instances = load_instances_from_directory(in_path)

    else:
        print('ERROR: path {} not found'.format(in_path))
        return

    total_time = 0
    total_fitness = 0
    total_value = 0
    unsolved_instances = 0
    for instance in instances:
        start_time = time.clock()
        solution = solve_genetic(instance)
        end_time = time.clock()
        time_taken = end_time - start_time
        total_time += time_taken
        total_fitness += solution.fitness()
        if solution.fitness() > 0:
            total_value += solution.value
        else:
            unsolved_instances += 1

        print('{} in {:.2f} s'.format(str(solution), time_taken))

    average_time = total_time / len(instances)
    print('Average time: {:.2f} s'.format(average_time))
    print('Unsolved instances: {}'.format(unsolved_instances))
    print('Average fitness: {:.2f}'.format(total_fitness / len(instances)))
    print('Total fitness: {}'.format(total_fitness))
    print('Average good solution value: {}'.format(total_value / (len(instances) - unsolved_instances)))


def load_instances_from_directory(directory):
    return [Instance.from_file(f) for f in glob.glob(os.path.join(directory, '*.inst.dat'))]


def main():
    cli()
