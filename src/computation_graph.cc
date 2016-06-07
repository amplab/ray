#include "computation_graph.h"

OperationId ComputationGraph::add_operation(std::unique_ptr<Operation> operation) {
  OperationId operationid = operations_.size();
  OperationId creator_operationid = operation->creator_operationid();
  if (spawned_operations_.size() != operationid) {
    QUARTZ_LOG(QUARTZ_FATAL, "ComputationGraph is attempting to call add_operation, but spawned_operations_.size() != operationid.");
  }
  operations_.emplace_back(std::move(operation));
  if (creator_operationid != NO_OPERATION && creator_operationid != ROOT_OPERATION) {
    spawned_operations_[creator_operationid].push_back(operationid);
  }
  spawned_operations_.push_back(std::vector<OperationId>());
  return operationid;
}

const Task& ComputationGraph::get_task(OperationId operationid) {
  if (operationid >= operations_.size()) {
    QUARTZ_LOG(QUARTZ_FATAL, "ComputationGraph attempting to get_task with operationid " << operationid << ", but operationid >= operations_.size().");
  }
  if (!operations_[operationid]->has_task()) {
    QUARTZ_LOG(QUARTZ_FATAL, "Calling get_task with operationid " << operationid << ", but this corresponds to a push not a task.");
  }
  return operations_[operationid]->task();
}
