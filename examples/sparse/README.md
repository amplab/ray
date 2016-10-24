# Some benchmarks for sparse matrix-vector products

Sparse matrix-vector products are the central operation of many sparse linear algebra
algorithms like conjugate gradient, Lanczos and more generally Krylov
subspace methods. These algorithms can for example be used to efficiently
solve sparse linear systems and compute eigenvalues of sparse matrices.
The PageRank algorithm is an example of a Krylov Subspace eigenvalue algorithm.
The runtime of Krylov subspace methods is almost exclusively determined by the
runtime of sparse matrix-vector multiplies, which makes the task and important
benchmark for distributed systems.

In this benchmark, we study how fast matrix-vector products can be evaluated in
parallel. We set up the benchmark in a way that is natural for Krylov subspace
methods: The initial vector and the result are required to be in the driver's
memory, the matrix is already distributed. This benchmark is challenging for
many data processing systems, because the ratio of computation to data is very
small and therefore system overhead like serialization might limit the amount
of parallelism.

As a test matrix, we use a symmetrized version of the matrix from
https://snap.stanford.edu/data/com-Orkut.html. It has shape (3 Mio, 3 Mio) and
about 230 Mio nonzero entries.

## Spark

For Spark, we run the following code in a pyspark shell:

```python
import cPickle
import numpy as np
import time

M = cPickle.load(open("...", "r"))
M = M[0:2000000,0:2000000]

num_partitions = 32

v = np.random.random(M.shape[1])

indices = np.array_split(np.arange(M.shape[0]), num_partitions)
M_partitioned = [M[indices[i],:] for i in range(0, num_partitions)]

v_remote = sc.broadcast(v);
matrix = sc.parallelize(range(num_partitions)).map(lambda i: M_partitioned[i]).cache()

a = time.time(); matrix.map(lambda M: M.dot(v)).reduce(lambda x, y: np.concatenate([x, y])); b = time.time() - a # warm up cache
a = time.time(); matrix.map(lambda M: M.dot(v)).reduce(lambda x, y: np.concatenate([x, y])); b = time.time() - a
```

The pyspark shell is started with:

```bash
./bin/pyspark --executor-cores 4 --executor-memory 80g
./bin/pyspark --executor-cores 4 --master ... --num-executors 2 --executor-memory 80g
```

## Dask

```python
from dask import delayed
import numpy as np
import cPickle
import time

M = cPickle.load(open("...", "r"))

num_partitions = 32

v = np.random.random(M.shape[1])

indices = np.array_split(np.arange(M.shape[0]), num_partitions)
M_partitioned = [M[indices[i],:] for i in range(0, num_partitions)]

@delayed
def mvp(M, v):
    return M.dot(v)

result = [mvp(partition, v) for partition in M_partitioned] # construct the task graph

a = time.time(); w = np.concatenate([r.compute() for r in result]); b = time.time() - a # single node dask (possibly cache)
a = time.time(); w = np.concatenate([r.compute() for r in result]); b = time.time() - a # single node dask

from distributed import Executor

executor = Executor('52.91.5.16:8786')

a = time.time(); l = executor.compute(result); np.concatenate([r.result() for r in l]); b = time.time() - a # dask.distributed (possibly cache)
a = time.time(); l = executor.compute(result); np.concatenate([r.result() for r in l]); b = time.time() - a # dask.distributed (possibly cache)
```

Here, dask.distributed was started using

```
dscheduler
dworker 127.0.0.1:8786
dworker 127.0.0.1:8786
dworker 127.0.0.1:8786
dworker 127.0.0.1:8786
```

and similar for the two node experiment.

## Halo

To run the Halo code, start the scheduler with

```
./scheduler 52.91.5.16:10001
```

and run `python mvp.py`.
