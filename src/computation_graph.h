#ifndef ORCHESTRA_COMPUTATIONGRAPH_H
#define ORCHESTRA_COMPUTATIONGRAPH_H

#include <iostream>
#include <limits>

#include "orchestra/orchestra.h"
#include "orchestra.grpc.pb.h"
#include "types.pb.h"

const OperationId NO_TASK = std::numeric_limits<TaskId>::max(); // used to represent the absence of a task

class ComputationGraph {
public:
  size_t num_tasks(); // return tasks_.size()
  // Add a task to the computation graph, this returns the TaskId for the new
  // task. This method takes ownership over task.
  TaskId add_task(std::unique_ptr<Operation> task);
  // Return the task corresponding to a particular TaskId. If taskid corresponds
  // to a push, then fail.
  const Task& get_task(OperationId taskid);
private:
  // maps a TaskId to the corresponding task or push
  std::vector<std::unique_ptr<TaskOrPush> > tasks_;
  // spawned_tasks_[taskid] is a vector of the TaskIds of the task calls and
  // push calls spawned by the task with TaskId taskid
  std::vector<std::vector<TaskId> > spawned_tasks_;
};

#endif
