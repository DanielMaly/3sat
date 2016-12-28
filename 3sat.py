import click
import math
import generator


@click.group()
def cli():
    """
    Entry point for the application's CLI
    """
    pass


@cli.command()
@click.option('--variables', '-v', type=click.INT, prompt='How many variables should the instances have?',
              help='The number of literals.')
@click.option('--count', '-c', default=1, help='Number of instances to generate.')
@click.option('--ratio', '-r', type=click.FLOAT, default=3.0, help='The ratio of clausules to variables')
def generate(variables, count, ratio):
    """
    Generates instances of the 3SAT problem.
    """
    clausules = math.floor(ratio * variables)
    for i in range(count):
        instance = generator.generate_instance(variables, clausules)
    pass