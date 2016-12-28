import click
import generator
import os


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
@click.option('--ratio', '-r', type=click.FLOAT, default=3.0, help='The ratio of clausules to variables')
@click.option('--weight-min', '-m', type=click.INT, default=1, help='The minimum clausule weight')
@click.option('--weight-max', '-x', type=click.INT, default=30, help='The maximum clausule weight')
@click.argument('out_directory')
def generate(literals, count, ratio, weight_min, weight_max, out_directory):
    """
    Generates instances of the 3SAT problem.
    """
    if not os.path.isdir(out_directory):
        print('ERROR: {} is not a directory'.format(out_directory))

    instances = [generator.generate_instance(literals, ratio, weight_min, weight_max) for i in range(count)]

    for i in range(count):
        with open(os.path.join(out_directory, '{0:03d}.inst.dat'.format(i)), mode='w') as out:
            out.write(str(instances[i]))


def main():
    cli()
