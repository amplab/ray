#ifndef ORCHESTRA_COMPUTATIONGRAPH_H
#define ORCHESTRA_COMPUTATIONGRAPH_H

#include <iostream>
#include <limits>

#include "orchestra/orchestra.h"
#include "orchestra.grpc.pb.h"
#include "types.pb.h"

const TaskId NO_TASK = std::numeric_limits<TaskId>::max(); // used to represent the absence of a task

class ComputationGraph {
public:
  size_t num_tasks(); // return tasks_.size()
  // Add a task to the computation graph, this returns the TaskId for the new
  // task. This method takes ownership over task.
  TaskId add_task(std::unique_ptr<TaskOrPush> task);
  // Return the task corresponding to a particular TaskId. If taskid corresponds
  // to a push, then fail.
  const Call& get_task(TaskId taskid);
  // Return the ids of the tasks spawned by the the task with id taskid.
  std::vector<TaskId> get_spawned_tasks(TaskId taskid);
  // If the objref corresponds to the output of a task, then return the id of
  // that task. If the objref corresponds to a pushed object, then return the id
  // of the task that invoked the push call. Note that it is possible that the
  // push call was invoked on the driver, in which case this returns NO_TASK.
  TaskId get_creator_taskid(ObjRef objref);
  TaskId get_spawned_taskid(TaskId creator_taskid, size_t spawned_task_index);
  std::vector<ObjRef> get_spawned_objrefs(TaskId creator_taskid, size_t spawned_task_index);
  bool is_new_task(TaskId creator_taskid, size_t spawned_task_index);
private:
  // maps a TaskId to the corresponding task or push
  std::vector<std::unique_ptr<TaskOrPush> > tasks_;
  // spawned_tasks_[taskid] is a vector of the TaskIds of the task calls and
  // push calls spawned by the task with TaskId taskid
  std::vector<std::vector<TaskId> > spawned_tasks_;
};

#endif
