## Installation

First, make sure you have Python 3 installed. The program has been developed and tested under Python 3.5, although slightly older versions of Python 3 will probably work too. You will also need a C compiler (tested with clang). 

To install the module, run the following command from the project root:

```
python3 setup.py develop
```

This should install dependencies and compile Cython files. 

## Running

You can run the program either in generator mode or in solver mode. To see available options for generating instances, run the following from the project root:

```
python . generate --help
```

NOTE: The directory specified in the OUT_DIRECTORY parameter must already exist before running the generator.

To run the solver, execute this command from the project root:

```
python . solve instances_dir
```

The ```instances_dir``` argument is where your problem instances in weighted DIMACS format are located. The solver will average results from all instances placed within the same directory.

