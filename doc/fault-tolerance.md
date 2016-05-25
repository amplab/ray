# Fault Tolerance

A Photon cluster may consist of hundreds or thousands of nodes, and so it is
important for Photon to gracefully handle the failure of some nodes. However, we
do not intend to design Photon to be robust to arbitrary failures. In
particular, we make the following assumptions:

1. We assume that the scheduler process does not fail.
2. We assume that the driver process does not fail.
3. We assume that the driver's local object store does not fail.

Furthermore, for fault tolerance to work, we need to assume that tasks are
deterministic.

## Design

The scheduler keeps track of a computation graph consisting of every task ever
sent to it. This graph includes the object references of the outputs of each
task and the object references of the inputs of each task.

The following data structures in the scheduler should still be correct:

1. `target_objrefs_`
2. `reverse_target_objrefs_`
3. `contained_objrefs_`: TODO(rkn): This will depend on whether control flow can
be stochastic or not.

The following data structures in the scheduler can be updated easily by removing
certain elements:

1. `workers_`: Remove the workers that died.
2. `objtable_`: Remove the object stores that died from every entry of
the `objtable_`.
3. `fntable_`: Remove the workers that died from every entry of `fntable_`.
4. `avail_workers_`: Remove the workers that died.

The following data structures in the scheduler will require more work to fix:

1. `reference_counts_`: If an object is lost from all object stores, decrement
the reference counts of the contained object references. Actually, the right
solution may be to reconstruct this data structure from scratch by polling all
of the workers and object stores and asking them for their reference counts. We
will have to figure out how to handle currently in-flight reference increments
and decrements.

task_queue_;
pull_queue_;
alias_notification_queue_;


## Steps in Fault Tolerance

1. The scheduler figures out that a failure has happened and figures out exactly
which workers and object stores have died. TODO(rkn): How should we do this?
Heartbeats?
2. The scheduler updates the data structures that are affected by the failures.
This will tell the scheduler which objects were lost (if there is some object
reference `objref` for which `objtable_[objref]` went from nonempty to empty,
then the corresponding object was lost.)
3. The scheduler adds the task of recomputing lost objects to `task_queue_`.
4. When the execution of a task spawns a new task, we return the object
references that were previously assigned to the outputs of that new task. We do
not assign new object references. If the objects corresponding to these object
references are still present, then we do not re-execute the task. If they are
not, then we add that task to the task queue.
5. The scheduler recovers objects that have died by adding tasks to the
task_queue_ to recover all of the lost objects. To elaborate on this,

    1. The scheduler determines which objects have been lost.
    2. The scheduler determines which tasks created those objects (either as
       the output of one task or via a push call).
    3. The scheduler reruns all of these tasks. However, these tasks are not
       allowed to spawn new tasks (because this step of fault tolerance may
       already be running those tasks).
6. Any task that was killed before it finished running is rerun and is allowed
to spawn new tasks. TODO(rkn): is it possible to lose an object that was created
by a currently running worker without losing that worker? Probably not at the
moment, but if we allow object migration later on then this could arise and
we'll have to handle it.


## Information Needed in Order to Recover

1. For each task, we must have the input object references, the output object
references, and the arguments that were passed by value.
2. For each task, we must keep track of a list of tasks invoked by the execution
of that task.

The scheduler maintains a data structure `tasks_`, which has type
`std::vector<Call>`, which is maps a `TaskId` to the corresponding `Call`.

The scheduler maintains a data structure `task_invocations_`, which has type
`std::vector<std::vector<TaskId> >`. The first vector is indexed by a `TaskId`,
and maps `taskid` to a vector of tasks that were invoked by that task.


## Details: Reference Counting

If we lose some object references due to failures (perhaps the object references
were Python objects on a worker or were serialized within an object in an object
store), should we decrement the counts for those object references? Doing so
will probably help avoid memory leaks. However, doing so may cause the objects
to be deallocated even though we may need them in the future.

Note that in order to appropriately decrement the reference counts, there are
two possibilities.

First, the scheduler could keep track of how many of each object reference are
on each worker. If a certain worker dies, then we can remove all of the object
references associated with that worker. The downside of this approach is that
the scheduler has to do more bookkeeping, and it will probably have to change if
we ever want to have scheduler fault tolerance. The upside of this approach is
that if a single machine fails, doing fault tolerance is more straightforward
and may not require any system-wide interruption.

Second, the scheduler could forget about its current count of object references,
and simply poll each worker and each object store and ask them for their current
counts and use their answers to reconstruct the system-wide counts. This second
option requires each worker to locally track its counts, which is
straightforward. The upside of this approach is that it will potentially be
compatible with supporting scheduler fault tolerance, and it places less burden
on the scheduler. The downside is that the failure of a single machine requires
a system-wide interruption. However, this may not be too much of a downside if
we don't think failures will happen very frequently.

## TODO

1. If objstore 1 is streaming an object to objstore 2 and objstore 1 dies part
of the way through, then it is the responsibility of objstore 2 to request
the delivery of the object again from the scheduler.
