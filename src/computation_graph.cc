#include "computationgraph.h"

TaskId ComputationGraph::num_tasks() {
  return tasks_.size();
}

TaskId ComputationGraph::add_task(std::unique_ptr<TaskOrPush> task) {
  TaskId taskid = tasks_.size();
  TaskId creator_taskid = task->creator_taskid();
  if (spawned_tasks_.size() != taskid) {
    ORCH_LOG(ORCH_FATAL, "ComputationGraph is attempting to add_task, but spawned_tasks_.size() != taskid.");
  }
  tasks_.emplace_back(std::move(task));
  if (creator_taskid != NO_TASK) {
    spawned_tasks_[creator_taskid].push_back(taskid);
  }
  spawned_tasks_.push_back(std::vector<TaskId>());
  return taskid;
}

const Call& ComputationGraph::get_task(TaskId taskid) {
  if (taskid >= tasks_.size()) {
    ORCH_LOG(ORCH_FATAL, "ComputationGraph attempting to get_task " << taskid << ", but taskid >= tasks_.size().");
  }
  if (!tasks_[taskid]->has_task()) {
    ORCH_LOG(ORCH_FATAL, "Calling get_task with taskid " << taskid << ", but this corresponds to a push not a task.");
  }
  return tasks_[taskid]->task();
}
