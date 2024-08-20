# Distributed-K-Means 
This repository contains a distributed implementation of the K-Means clustering algorithm using mpi4py. The implementation leverages the Message Passing Interface (MPI) to distribute the computational workload across multiple processors, enabling efficient clustering of large datasets.

## Prerequisites

Before running the code, ensure you have the following installed:

- **Python**: Version 3.x
- **mpi4py**: Python bindings for MPI

### Installing `mpi4py`

You can install `mpi4py` using `pip`:

```bash
python -m pip install mpi4py
```` 

###Usage
```bash
mpirun -np <number_of_processes> python kmeans.py
