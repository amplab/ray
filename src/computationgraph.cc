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
  if (creator_tasks_.size() != taskid) {
    ORCH_LOG(ORCH_FATAL, "ComputationGraph is attempting to add_task, but creator_tasks_.size() != taskid.");
  }
  tasks_.emplace_back(std::move(task));
  creator_tasks_.push_back(creator_taskid); // TODO(rkn): Get rid of this.
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

std::vector<TaskId> ComputationGraph::get_spawned_tasks(TaskId taskid) {
  return spawned_tasks_[taskid];
}

// TODO(rkn): Make this more efficient, probably by storing the information in a
// data structure.
TaskId ComputationGraph::get_creator_taskid(ObjRef objref) {
  for (int taskid = 0; taskid < tasks_.size(); ++taskid) {
    if (tasks_[taskid]->has_task()) {
      for (int i = 0; i < tasks_[taskid]->task().result_size(); ++i) {
        if (tasks_[taskid]->task().result(i) == objref) {
          return taskid;
        }
      }
    } else {
      if (tasks_[taskid]->push().objref() == objref) {
        return tasks_[taskid]->creator_taskid();
      }
    }
  }
  ORCH_LOG(ORCH_FATAL, "Unable to find the task that created objref " << objref);
}

// TODO(rkn): Should this really take a creator_taskid? Or should it take a taskid?
TaskId ComputationGraph::get_spawned_taskid(TaskId creator_taskid, size_t spawned_task_index) {
  // TODO(rkn): What if creator_taskid == NO_TASK
  return spawned_tasks_[creator_taskid][spawned_task_index];
}

// TODO(rkn): Should this really take a creator_taskid? Or should it take a taskid?
std::vector<ObjRef> ComputationGraph::get_spawned_objrefs(TaskId creator_taskid, size_t spawned_task_index) {
  if (!tasks_[creator_taskid]->has_task()) {
    ORCH_LOG(ORCH_FATAL, "Calling get_spawned_objrefs on task " << creator_taskid << ", but this is a push call.");
  }
  TaskId taskid = spawned_tasks_[creator_taskid][spawned_task_index];
  std::vector<ObjRef> objrefs;
  for (int i = 0; i < tasks_[taskid]->task().result_size(); ++i) {
    objrefs.push_back(tasks_[taskid]->task().result(i));
  }
  return objrefs;
}

bool ComputationGraph::is_new_task(TaskId creator_taskid, size_t spawned_task_index) {
  if (creator_taskid == NO_TASK) {
    return true;
  }
  if (spawned_task_index >= spawned_tasks_[creator_taskid].size()) {
    return true;
  }
  return false;
}
