#include "scheduler.h"

#include <random>
#include <thread>
#include <chrono>

#include "utils.h"

SchedulerService::SchedulerService(SchedulingAlgorithmType scheduling_algorithm) : scheduling_algorithm_(scheduling_algorithm) {}

Status SchedulerService::RemoteCall(ServerContext* context, const RemoteCallRequest* request, RemoteCallReply* reply) {
  auto temp_task = request->call().New();
  // auto temp_task = std::unique_ptr<Call>(request->call().New());
  temp_task->CopyFrom(request->call());

  fntable_lock_.lock();
  // ORCH_LOG(ORCH_DEBUG, "In RemoteCall: Acquired fntable_lock_");
  if (fntable_.find(temp_task->name()) == fntable_.end()) {
    // TODO(rkn): In the future, this should probably not be fatal. Instead, propagate the error back to the worker.
    ORCH_LOG(ORCH_FATAL, "The function " << temp_task->name() << " has not been registered by any worker.");
  }
  size_t num_return_vals = fntable_[temp_task->name()].num_return_vals();
  fntable_lock_.unlock();
  // ORCH_LOG(ORCH_DEBUG, "In RemoteCall: Released fntable_lock_");

  computation_graph_lock_.lock();
  current_tasks_lock_.lock();

  std::pair<TaskId, std::pair<bool, size_t> > current_task_info = current_tasks_[request->workerid()];
  TaskId creator_taskid = current_task_info.first;
  bool creator_task_is_new = current_task_info.second.first;
  size_t spawned_task_index = current_task_info.second.second;
  current_tasks_[request->workerid()].second.second += 1;
  bool current_task_is_new = computation_graph_.is_new_task(creator_taskid, spawned_task_index);

  TaskId taskid;
  std::vector<ObjRef> result_objrefs;
  if (current_task_is_new) {
    // this is a new task
    for (size_t i = 0; i < num_return_vals; ++i) {
      ObjRef result = register_new_object();
      temp_task->add_result(result);
      // temp_task->add_store_output(true);
      result_objrefs.push_back(result);
    }
    auto task_or_push = std::unique_ptr<TaskOrPush>(new TaskOrPush());
    task_or_push->set_allocated_task(temp_task);
    // task_or_push->mutable_task()->CopyFrom(temp_task);
    task_or_push->set_creator_taskid(creator_taskid);
    taskid = computation_graph_.add_task(std::move(task_or_push));
    // taskid = computation_graph_.add_task(std::move(temp_task), creator_taskid);
  } else {
    // This task is not new, so do not run it. When we rerun tasks for fault
    // tolerance reasons, we do not allow the tasks to spawn new tasks.
    taskid = computation_graph_.get_spawned_taskid(creator_taskid, spawned_task_index);
    ORCH_LOG(ORCH_DEBUG, "In RemoteCall, not rerunning task " << taskid << " because it is not new.");
    result_objrefs = computation_graph_.get_spawned_objrefs(creator_taskid, spawned_task_index);
    if (result_objrefs.size() != num_return_vals) {
      // ORCH_LOG(ORCH_DEBUG, "creator_taskid = " << creator_taskid);
      // ORCH_LOG(ORCH_DEBUG, "spawned_task_index = " << spawned_task_index);
      // ORCH_LOG(ORCH_DEBUG, "result_objrefs.size() = " << result_objrefs.size());
      // ORCH_LOG(ORCH_DEBUG, "num_return_vals = " << num_return_vals);
      ORCH_LOG(ORCH_FATAL, "Attempting to rerun task " << taskid << ", but result_objrefs.size() != num_return_vals.");
      // TODO(rkn): Check that arguments passed by value and ref are the same as what we have cached.
    }
  }
  const Call& task = computation_graph_.get_task(taskid);
  computation_graph_lock_.unlock();
  current_tasks_lock_.unlock();

  /*
  task.clear_store_output();
  for (int i = 0; i < num_return_vals; ++i) {
    task.add_store_output(!already_present(task.result(i)));
  }
  */

  if (num_return_vals != result_objrefs.size()) {
    ORCH_LOG(ORCH_FATAL, "In Scheduler::RemoteCall, num_return_vals != result_objrefs.size()");
  }
  for (size_t i = 0; i < num_return_vals; ++i) {
    reply->add_result(result_objrefs[i]);
  }
  {
    std::lock_guard<std::mutex> reference_counts_lock(reference_counts_lock_); // we grab this lock because increment_ref_count assumes it has been acquired
    increment_ref_count(result_objrefs); // We increment once so the objrefs don't go out of scope before we reply to the worker that called RemoteCall. The corresponding decrement will happen in remote_call in orchpylib.
    // TODO(rkn): Maybe we don't want the increment below if !current_task_is_new because we don't do the decrement in deserialize_call. Think about this.
    increment_ref_count(result_objrefs); // We increment once so the objrefs don't go out of scope before the task is scheduled on the worker. The corresponding decrement will happen in deserialize_call in orchpylib.
  }

  if (current_task_is_new) {
    // ORCH_LOG(ORCH_DEBUG, "In RemoteCall: Acquiring task_queue");
    task_queue_lock_.lock();
    task_queue_.push_back(std::make_pair(taskid, current_task_is_new));
    task_queue_lock_.unlock();
    // ORCH_LOG(ORCH_DEBUG, "In RemoteCall: Released task_queue");
  }

  schedule();
  return Status::OK;
}

Status SchedulerService::PushObj(ServerContext* context, const PushObjRequest* request, PushObjReply* reply) {
  // TODO(rkn): We cannot re-execute a push that was done by the driver. Should we check for that case here?
  computation_graph_lock_.lock();
  current_tasks_lock_.lock();

  std::pair<TaskId, std::pair<bool, size_t> > current_task_info = current_tasks_[request->workerid()];
  TaskId creator_taskid = current_task_info.first;
  bool creator_task_is_new = current_task_info.second.first;
  size_t spawned_task_index = current_task_info.second.second;
  current_tasks_[request->workerid()].second.second += 1;
  bool current_task_is_new = computation_graph_.is_new_task(creator_taskid, spawned_task_index);

  ObjRef objref = 0;
  if (current_task_is_new) {
    // this is a new push
    objref = register_new_object();
    reply->set_objref(objref);
    reply->set_already_present(false);

    Push* push = new Push();
    push->set_objref(objref);
    auto push_task = std::unique_ptr<TaskOrPush>(new TaskOrPush());
    push_task->set_allocated_push(push);
    push_task->set_creator_taskid(creator_taskid);
    computation_graph_.add_task(std::move(push_task));
    // auto push_task = std::unique_ptr<Call>(new Call());
    // push_task->add_result(objref);
    // computation_graph_.add_task(std::move(push_task), creator_taskid);

    // Call push_task;
    // push_task.add_result(objref); // TODO(rkn): Do we want to add any other information like the name "push"?
    // computation_graph_.add_task(&push_task, creator_taskid);
  }
  else {
    // this is a push that we are re-running for fault tolerance reasons
    ObjRef objref = computation_graph_.get_spawned_objrefs(creator_taskid, spawned_task_index)[0];
    reply->set_objref(objref);
    bool object_still_present = false;
    reply->set_already_present(already_present(objref));
    if (has_canonical_objref(objref)) {
      if (!is_canonical(objref)) {
        ORCH_LOG(ORCH_FATAL, "objref was the result of a push call, but !is_canonical(objref).");
      }
      objtable_lock_.lock();
      object_still_present = (objtable_[objref].size() > 0);
      objtable_lock_.unlock();
    }
    if (object_still_present) {
      reply->set_already_present(true);
      ORCH_LOG(ORCH_FATAL, "In PushObj, re-running push of objref " << objref << " from creator task " << creator_taskid << ", the object is still in an object store."); // TODO(rkn): Later this shouldn't be fatal, but right now fault tolerance should never be triggered.
    }
    else {
      reply->set_already_present(false);
      ORCH_LOG(ORCH_FATAL, "In PushObj, re-running push of objref " << objref << " from creator task " << creator_taskid << ", the object is no longer in an object store."); // TODO(rkn): Later this shouldn't be fatal, but right now fault tolerance should never be triggered.
    }
  }

  computation_graph_lock_.unlock();
  current_tasks_lock_.unlock();

  schedule();
  return Status::OK;
}

Status SchedulerService::RequestObj(ServerContext* context, const RequestObjRequest* request, AckReply* reply) {
  objtable_lock_.lock();
  size_t size = objtable_.size();
  objtable_lock_.unlock();

  ObjRef objref = request->objref();
  if (objref >= size) {
    ORCH_LOG(ORCH_FATAL, "internal error: no object with objref " << objref << " exists");
  }

  pull_queue_lock_.lock();
  pull_queue_.push_back(std::make_pair(request->workerid(), objref));
  pull_queue_lock_.unlock();
  schedule();
  return Status::OK;
}

Status SchedulerService::AliasObjRefs(ServerContext* context, const AliasObjRefsRequest* request, AckReply* reply) {
  ObjRef alias_objref = request->alias_objref();
  ObjRef target_objref = request->target_objref();
  ORCH_LOG(ORCH_ALIAS, "Aliasing objref " << alias_objref << " with objref " << target_objref);
  if (alias_objref == target_objref) {
    ORCH_LOG(ORCH_FATAL, "internal error: attempting to alias objref " << alias_objref << " with itself.");
  }
  objtable_lock_.lock();
  size_t size = objtable_.size();
  objtable_lock_.unlock();
  if (alias_objref >= size) {
    ORCH_LOG(ORCH_FATAL, "internal error: no object with objref " << alias_objref << " exists");
  }
  if (target_objref >= size) {
    ORCH_LOG(ORCH_FATAL, "internal error: no object with objref " << target_objref << " exists");
  }
  {
    std::lock_guard<std::mutex> target_objrefs_lock(target_objrefs_lock_);
    if (target_objrefs_[alias_objref] != UNITIALIZED_ALIAS) {
      ORCH_LOG(ORCH_FATAL, "internal error: attempting to alias objref " << alias_objref << " with objref " << target_objref << ", but objref " << alias_objref << " has already been aliased with objref " << target_objrefs_[alias_objref]);
    }
    target_objrefs_[alias_objref] = target_objref;
  }
  {
    std::lock_guard<std::mutex> reverse_target_objrefs_lock(reverse_target_objrefs_lock_);
    reverse_target_objrefs_[target_objref].push_back(alias_objref);
  }
  schedule();
  return Status::OK;
}

Status SchedulerService::RegisterObjStore(ServerContext* context, const RegisterObjStoreRequest* request, RegisterObjStoreReply* reply) {
  std::lock_guard<std::mutex> objstore_lock(objstores_lock_);
  ObjStoreId objstoreid = objstores_.size();
  auto channel = grpc::CreateChannel(request->objstore_address(), grpc::InsecureChannelCredentials());
  objstores_.push_back(ObjStoreHandle());
  objstores_[objstoreid].alive = true;
  objstores_[objstoreid].address = request->objstore_address();
  objstores_[objstoreid].channel = channel;
  objstores_[objstoreid].objstore_stub = ObjStore::NewStub(channel);
  reply->set_objstoreid(objstoreid);
  return Status::OK;
}

Status SchedulerService::RegisterWorker(ServerContext* context, const RegisterWorkerRequest* request, RegisterWorkerReply* reply) {
  std::pair<WorkerId, ObjStoreId> info = register_worker(request->worker_address(), request->objstore_address());
  WorkerId workerid = info.first;
  ObjStoreId objstoreid = info.second;
  ORCH_LOG(ORCH_INFO, "registered worker with workerid " << workerid);
  reply->set_workerid(workerid);
  reply->set_objstoreid(objstoreid);
  schedule();
  return Status::OK;
}

Status SchedulerService::RegisterFunction(ServerContext* context, const RegisterFunctionRequest* request, AckReply* reply) {
  ORCH_LOG(ORCH_INFO, "register function " << request->fnname() <<  " from workerid " << request->workerid());
  register_function(request->fnname(), request->workerid(), request->num_return_vals());
  schedule();
  return Status::OK;
}

Status SchedulerService::ObjReady(ServerContext* context, const ObjReadyRequest* request, AckReply* reply) {
  ObjRef objref = request->objref();
  ORCH_LOG(ORCH_DEBUG, "object " << objref << " ready on store " << request->objstoreid());
  add_canonical_objref(objref);
  add_location(objref, request->objstoreid());
  schedule();
  return Status::OK;
}

Status SchedulerService::WorkerReady(ServerContext* context, const WorkerReadyRequest* request, AckReply* reply) {
  WorkerId workerid = request->workerid();
  ORCH_LOG(ORCH_INFO, "worker " << workerid << " reported back");
  {
    std::lock_guard<std::mutex> lock(workers_lock_);
    if (!workers_[workerid].alive) {
      ORCH_LOG(ORCH_DEBUG, "workerid " << workerid << " just called WorkerReady, but workers_[workerid].alive == false. This is probably ok but could be an error.");
      return Status::OK;
    }
  }
  {
    // Here we should update avail_workers_ and current_tasks_ at the same time.
    // Updating them seperately caused a bug in the past.
    std::lock_guard<std::mutex> avail_workers_lock(avail_workers_lock_);
    std::lock_guard<std::mutex> current_tasks_lock(current_tasks_lock_);
    if (current_tasks_[workerid].first == NO_TASK) {
      ORCH_LOG(ORCH_FATAL, "workerid " << workerid << " just called WorkerReady, but current_tasks[workerid].first == NO_TASK");
    }
    current_tasks_[workerid].first = NO_TASK;
    current_tasks_[workerid].second.first = false;
    current_tasks_[workerid].second.second = 0;
    avail_workers_.push_back(workerid);
  }
  schedule();
  return Status::OK;
}

Status SchedulerService::IncrementRefCount(ServerContext* context, const IncrementRefCountRequest* request, AckReply* reply) {
  int num_objrefs = request->objref_size();
  if (num_objrefs == 0) {
    ORCH_LOG(ORCH_FATAL, "Scheduler received IncrementRefCountRequest with 0 objrefs.");
  }
  std::vector<ObjRef> objrefs;
  for (int i = 0; i < num_objrefs; ++i) {
    objrefs.push_back(request->objref(i));
  }
  std::lock_guard<std::mutex> reference_counts_lock(reference_counts_lock_); // we grab this lock because increment_ref_count assumes it has been acquired
  increment_ref_count(objrefs);
  return Status::OK;
}

Status SchedulerService::DecrementRefCount(ServerContext* context, const DecrementRefCountRequest* request, AckReply* reply) {
  int num_objrefs = request->objref_size();
  if (num_objrefs == 0) {
    ORCH_LOG(ORCH_FATAL, "Scheduler received DecrementRefCountRequest with 0 objrefs.");
  }
  std::vector<ObjRef> objrefs;
  for (int i = 0; i < num_objrefs; ++i) {
    objrefs.push_back(request->objref(i));
  }
  std::lock_guard<std::mutex> reference_counts_lock(reference_counts_lock_); // we grab this lock, because decrement_ref_count assumes it has been acquired
  decrement_ref_count(objrefs);
  return Status::OK;
}

Status SchedulerService::AddContainedObjRefs(ServerContext* context, const AddContainedObjRefsRequest* request, AckReply* reply) {
  ObjRef objref = request->objref();
  // if (!is_canonical(objref)) {
    // TODO(rkn): Perhaps we don't need this check. It won't work because the objstore may not have called ObjReady yet.
    // ORCH_LOG(ORCH_FATAL, "Attempting to add contained objrefs for non-canonical objref " << objref);
  // }
  std::lock_guard<std::mutex> contained_objrefs_lock(contained_objrefs_lock_);
  if (contained_objrefs_[objref].size() != 0) {
    ORCH_LOG(ORCH_FATAL, "Attempting to add contained objrefs for objref " << objref << ", but contained_objrefs_[objref].size() != 0.");
  }
  for (int i = 0; i < request->contained_objref_size(); ++i) {
    contained_objrefs_[objref].push_back(request->contained_objref(i));
  }
  return Status::OK;
}

Status SchedulerService::SchedulerInfo(ServerContext* context, const SchedulerInfoRequest* request, SchedulerInfoReply* reply) {
  get_info(*request, reply);
  return Status::OK;
}

Status SchedulerService::KillObjStore(ServerContext* context, const KillObjStoreRequest* request, AckReply* reply) {
  ObjStoreId objstoreid = request->objstoreid();
  ORCH_LOG(ORCH_INFO, "Scheduler is attempting to terminate object store " << objstoreid);
  objstores_lock_.lock();
  kill_objstore(objstoreid);
  objstores_lock_.unlock();
  // TODO(rkn): Insert random wait here? Does it matter?
  recover_from_failed_objstore(objstoreid);
  return Status::OK;
}

Status SchedulerService::KillWorker(ServerContext* context, const KillWorkerRequest* request, AckReply* reply) {
  WorkerId workerid = request->workerid();
  ORCH_LOG(ORCH_INFO, "Scheduler is attempting to terminate worker " << workerid);
  workers_lock_.lock();
  kill_worker(workerid);
  workers_lock_.unlock();
  return Status::OK;
}

// TODO(rkn): This could execute multiple times with the same arguments before
// the delivery finishes, but we only want it to happen once. Currently, the
// redundancy is handled by the object store, which will only execute the
// delivery once. However, we may want to handle it in the scheduler in the
// future.
//
// deliver_object assumes that the aliasing for objref has already been completed. That is, has_canonical_objref(objref) == true
// deliver_object returns a bool. True indicates that the delivery was
// successfully started. False indicates that the delivery did not successfully
// start.
bool SchedulerService::deliver_object(ObjRef objref, ObjStoreId from, ObjStoreId to) {
  if (from == to) {
    ORCH_LOG(ORCH_FATAL, "attempting to deliver objref " << objref << " from objstore " << from << " to itself.");
  }
  if (!has_canonical_objref(objref)) {
    ORCH_LOG(ORCH_FATAL, "attempting to deliver objref " << objref << ", but this objref does not yet have a canonical objref.");
  }
  if (!objstores_[from].alive) {
    ORCH_LOG(ORCH_INFO, "attempting to deliver objref " << objref << " from objstore " << from << " to objstore " << to << ", but objstore " << from << " is not alive.");
    return false;
  }
  if (!objstores_[to].alive) {
    ORCH_LOG(ORCH_INFO, "attempting to deliver objref " << objref << " from objstore " << from << " to objstore " << to << ", but objstore " << to << " is not alive.");
    return false;
  }
  ClientContext context;
  AckReply reply;
  StartDeliveryRequest request;
  ObjRef canonical_objref = get_canonical_objref(objref);
  request.set_objref(canonical_objref);
  std::lock_guard<std::mutex> lock(objstores_lock_);
  request.set_objstore_address(objstores_[from].address);
  objstores_[to].objstore_stub->StartDelivery(&context, request, &reply);
  return true; // TODO(rkn): Actually parse the reply and see if it succeeded or failed.
}

void SchedulerService::schedule() {
  // TODO(rkn): Do this more intelligently.
  perform_pulls(); // See what we can do in pull_queue_
  if (scheduling_algorithm_ == SCHEDULING_ALGORITHM_NAIVE) {
    schedule_tasks_naively(); // See what we can do in task_queue_
  } else if (scheduling_algorithm_ == SCHEDULING_ALGORITHM_LOCALITY_AWARE) {
    schedule_tasks_location_aware(); // See what we can do in task_queue_
  } else {
    ORCH_LOG(ORCH_FATAL, "scheduling algorithm not known");
  }
  perform_notify_aliases(); // See what we can do in alias_notification_queue_
}

// submit task assumes that the canonical objrefs for its arguments are all
// ready, that is has_canonical_objref() is true for all of the call's arguments
// This method returns true if the task was successfully submitted and false if
// the task was not successfully submitted.
bool SchedulerService::submit_task(TaskId taskid, const Call& call, bool is_new_task, WorkerId workerid) {
  // computation_graph_lock_.lock();
  // Call* call = computation_graph_.get_task(taskid);
  // computation_graph_lock_.unlock();
  ClientContext context;
  InvokeCallRequest request;
  InvokeCallReply reply;
  ORCH_LOG(ORCH_INFO, "Scheduler attempting to submit_task " << taskid << " to worker " << workerid << ".");
  ORCH_LOG(ORCH_INFO, "starting to send arguments");
  for (size_t i = 0; i < call.arg_size(); ++i) {
    if (!call.arg(i).has_obj()) {
      ObjRef objref = call.arg(i).ref();
      // ORCH_LOG(ORCH_DEBUG, "    IN SUBMIT_TASK: AAAA");
      ObjRef canonical_objref = get_canonical_objref(objref);
      {
        // Notify the relevant objstore about potential aliasing when it's ready
        // ORCH_LOG(ORCH_DEBUG, "    IN SUBMIT_TASK: BBBB");
        std::lock_guard<std::mutex> alias_notification_queue_lock(alias_notification_queue_lock_);
        alias_notification_queue_.push_back(std::make_pair(get_store(workerid), std::make_pair(objref, canonical_objref)));
      }
      // ORCH_LOG(ORCH_DEBUG, "    IN SUBMIT_TASK: CCCC");
      attempt_notify_alias(get_store(workerid), objref, canonical_objref);

      // ORCH_LOG(ORCH_DEBUG, "call contains object ref " << canonical_objref);
      // ORCH_LOG(ORCH_DEBUG, "    IN SUBMIT_TASK: DDDD");
      std::lock_guard<std::mutex> objtable_lock(objtable_lock_);
      auto &objstores = objtable_[canonical_objref];
      // ORCH_LOG(ORCH_DEBUG, "    IN SUBMIT_TASK: EEEE");
      std::lock_guard<std::mutex> workers_lock(workers_lock_);
      if (!std::binary_search(objstores.begin(), objstores.end(), workers_[workerid].objstoreid)) { // TODO(rkn): replace this with get_store
        // ORCH_LOG(ORCH_DEBUG, "    IN SUBMIT_TASK: FFFF");
        if (!deliver_object(canonical_objref, pick_objstore(canonical_objref), workers_[workerid].objstoreid)) { // TODO(rkn): replace this with get_store
          return false; // If the scheduler fails to initiate one of the deliveries, then submit_task has failed.
        }
      }
      // ORCH_LOG(ORCH_DEBUG, "    IN SUBMIT_TASK: GGGG");
    }
    // ORCH_LOG(ORCH_DEBUG, "    IN SUBMIT_TASK: HHHH");
  }

  // ORCH_LOG(ORCH_DEBUG, "    IN SUBMIT_TASK: IIII");
  current_tasks_lock_.lock();
  // ORCH_LOG(ORCH_DEBUG, "    IN SUBMIT_TASK: IIII: ARE WE GETTING HERE?");
  if (current_tasks_[workerid].first != NO_TASK) {
    ORCH_LOG(ORCH_FATAL, "Attempting to submit_task to worker " << workerid << ", but current_tasks_[workerid].first != NO_TASK.");
  }
  current_tasks_[workerid].first = taskid;
  current_tasks_[workerid].second.first = is_new_task;
  current_tasks_[workerid].second.second = 0;
  current_tasks_lock_.unlock();
  // ORCH_LOG(ORCH_DEBUG, "    IN SUBMIT_TASK: IIII");

  CallToExecute call_to_execute;
  call_to_execute.mutable_call()->CopyFrom(call);
  call_to_execute.set_first_execution(is_new_task);
  for (int i = 0; i < call.result_size(); ++i) {
    call_to_execute.add_store_output(!already_present(call.result(i)));
    if (is_new_task && already_present(call.result(i))) {
      ORCH_LOG(ORCH_FATAL, "Task " << taskid << " is new, but objref " << call.result(i) << " is already present.");
    }
  }

  request.mutable_call()->CopyFrom(call_to_execute);

  // ORCH_LOG(ORCH_DEBUG, "IN SUBMIT_TASK: call_to_execute.call().name() = " << request.call().call().name());

  // ORCH_LOG(ORCH_DEBUG, "    IN SUBMIT_TASK: JJJJ");
  workers_lock_.lock();
  if (!workers_[workerid].alive) {
    ORCH_LOG(ORCH_FATAL, "Scheduler attempting to submit_task to worker " << workerid << ", but this worker is no longer alive.");
  }
  Status status = workers_[workerid].worker_stub->InvokeCall(&context, request, &reply); // TODO(rkn): We shouldn't be holding on to workers_lock_ while doing a GRPC call.

  // ORCH_LOG(ORCH_DEBUG, "IN SUBMIT_TASK: call_to_execute.call().name() = " << request.call().call().name());

  workers_lock_.unlock();
  // ORCH_LOG(ORCH_DEBUG, "    IN SUBMIT_TASK: KKKK");
  return true;
}

bool SchedulerService::can_run(const Call& task) {
  std::lock_guard<std::mutex> lock(objtable_lock_);
  for (int i = 0; i < task.arg_size(); ++i) {
    if (!task.arg(i).has_obj()) {
      ObjRef objref = task.arg(i).ref();
      if (!has_canonical_objref(objref)) {
        return false;
      }
      ObjRef canonical_objref = get_canonical_objref(objref);
      if (canonical_objref >= objtable_.size() || objtable_[canonical_objref].size() == 0) {
        return false;
      }
    }
  }
  return true;
}

std::pair<WorkerId, ObjStoreId> SchedulerService::register_worker(const std::string& worker_address, const std::string& objstore_address) {
  ORCH_LOG(ORCH_INFO, "registering worker " << worker_address << " connected to object store " << objstore_address);
  ObjStoreId objstoreid = std::numeric_limits<size_t>::max();
  for (int num_attempts = 0; num_attempts < 5; ++num_attempts) {
    std::lock_guard<std::mutex> lock(objstores_lock_);
    for (size_t i = 0; i < objstores_.size(); ++i) {
      if (objstores_[i].address == objstore_address) {
        objstoreid = i;
      }
    }
    if (objstoreid == std::numeric_limits<size_t>::max()) {
      std::this_thread::sleep_for (std::chrono::milliseconds(100));
    }
  }
  if (objstoreid == std::numeric_limits<size_t>::max()) {
    ORCH_LOG(ORCH_FATAL, "object store with address " << objstore_address << " not yet registered");
  }
  // current_tasks_ and avail_workers_ need to be updated at the same time.
  std::lock_guard<std::mutex> current_tasks_lock(current_tasks_lock_);
  std::lock_guard<std::mutex> avail_workers_lock(avail_workers_lock_);
  std::lock_guard<std::mutex> workers_lock(workers_lock_);
  WorkerId workerid = workers_.size();
  if (current_tasks_.size() != workerid) {
    ORCH_LOG(ORCH_FATAL, "Attempting to register_worker, but workers_.size() != current_tasks_.size().");
  }
  workers_.push_back(WorkerHandle());
  auto channel = grpc::CreateChannel(worker_address, grpc::InsecureChannelCredentials());
  workers_[workerid].alive = true;
  workers_[workerid].channel = channel;
  workers_[workerid].objstoreid = objstoreid;
  workers_[workerid].worker_stub = WorkerService::NewStub(channel);
  current_tasks_.push_back(std::make_pair(NO_TASK, std::make_pair(false, 0)));
  avail_workers_.push_back(workerid);

  return std::make_pair(workerid, objstoreid);
}

ObjRef SchedulerService::register_new_object() {
  // If we don't simultaneously lock objtable_ and target_objrefs_, we will probably get errors.
  // TODO(rkn): increment/decrement_reference_count also acquire reference_counts_lock_ and target_objrefs_lock_ (through has_canonical_objref()), which caused deadlock in the past
  std::lock_guard<std::mutex> reference_counts_lock(reference_counts_lock_);
  std::lock_guard<std::mutex> contained_objrefs_lock(contained_objrefs_lock_);
  std::lock_guard<std::mutex> objtable_lock(objtable_lock_);
  std::lock_guard<std::mutex> target_objrefs_lock(target_objrefs_lock_);
  std::lock_guard<std::mutex> reverse_target_objrefs_lock(reverse_target_objrefs_lock_);
  ObjRef objtable_size = objtable_.size();
  ObjRef target_objrefs_size = target_objrefs_.size();
  ObjRef reverse_target_objrefs_size = reverse_target_objrefs_.size();
  ObjRef reference_counts_size = reference_counts_.size();
  ObjRef contained_objrefs_size = contained_objrefs_.size();
  if (objtable_size != target_objrefs_size) {
    ORCH_LOG(ORCH_FATAL, "objtable_ and target_objrefs_ should have the same size, but objtable_.size() = " << objtable_size << " and target_objrefs_.size() = " << target_objrefs_size);
  }
  if (objtable_size != reverse_target_objrefs_size) {
    ORCH_LOG(ORCH_FATAL, "objtable_ and reverse_target_objrefs_ should have the same size, but objtable_.size() = " << objtable_size << " and reverse_target_objrefs_.size() = " << reverse_target_objrefs_size);
  }
  if (objtable_size != reference_counts_size) {
    ORCH_LOG(ORCH_FATAL, "objtable_ and reference_counts_ should have the same size, but objtable_.size() = " << objtable_size << " and reference_counts_.size() = " << reference_counts_size);
  }
  if (objtable_size != contained_objrefs_size) {
    ORCH_LOG(ORCH_FATAL, "objtable_ and contained_objrefs_ should have the same size, but objtable_.size() = " << objtable_size << " and contained_objrefs_.size() = " << contained_objrefs_size);
  }
  objtable_.push_back(std::vector<ObjStoreId>());
  target_objrefs_.push_back(UNITIALIZED_ALIAS);
  reverse_target_objrefs_.push_back(std::vector<ObjRef>());
  reference_counts_.push_back(0);
  contained_objrefs_.push_back(std::vector<ObjRef>());
  return objtable_size;
}

void SchedulerService::add_location(ObjRef canonical_objref, ObjStoreId objstoreid) {
  // add_location must be called with a canonical objref
  if (!is_canonical(canonical_objref)) {
    ORCH_LOG(ORCH_FATAL, "Attempting to call add_location with a non-canonical objref (objref " << canonical_objref << ")");
  }
  std::lock_guard<std::mutex> objtable_lock(objtable_lock_);
  if (canonical_objref >= objtable_.size()) {
    ORCH_LOG(ORCH_FATAL, "trying to put an object in the object store that was not registered with the scheduler (objref " << canonical_objref << ")");
  }
  if (std::binary_search(objtable_[canonical_objref].begin(), objtable_[canonical_objref].end(), objstoreid)) {
    ORCH_LOG(ORCH_FATAL, "attempting to add objstore " << objstoreid << " to objtable for canonical_objref " << canonical_objref << ", but the objstore is already present in objtable_.");
  }
  // do a binary search
  auto pos = std::lower_bound(objtable_[canonical_objref].begin(), objtable_[canonical_objref].end(), objstoreid);
  if (pos == objtable_[canonical_objref].end() || objstoreid < *pos) {
    objtable_[canonical_objref].insert(pos, objstoreid);
  }
}

void SchedulerService::add_canonical_objref(ObjRef objref) {
  std::lock_guard<std::mutex> lock(target_objrefs_lock_);
  if (objref >= target_objrefs_.size()) {
    ORCH_LOG(ORCH_FATAL, "internal error: attempting to insert objref " << objref << " in target_objrefs_, but target_objrefs_.size() is " << target_objrefs_.size());
  }
  if (target_objrefs_[objref] != UNITIALIZED_ALIAS && target_objrefs_[objref] != objref) {
    ORCH_LOG(ORCH_FATAL, "internal error: attempting to declare objref " << objref << " as a canonical objref, but target_objrefs_[objref] is already aliased with objref " << target_objrefs_[objref]);
  }
  target_objrefs_[objref] = objref;
}

ObjStoreId SchedulerService::get_store(WorkerId workerid) {
  std::lock_guard<std::mutex> lock(workers_lock_);
  if (!workers_[workerid].alive) {
    ORCH_LOG(ORCH_FATAL, "Scheduler attempting to call get_store on worker " << workerid << ", but this worker is no longer alive.");
  }
  ObjStoreId result = workers_[workerid].objstoreid;
  return result;
}

void SchedulerService::register_function(const std::string& name, WorkerId workerid, size_t num_return_vals) {
  std::lock_guard<std::mutex> lock(fntable_lock_);
  FnInfo& info = fntable_[name];
  info.set_num_return_vals(num_return_vals);
  info.add_worker(workerid);
}

void SchedulerService::get_info(const SchedulerInfoRequest& request, SchedulerInfoReply* reply) {
  acquire_all_locks();
  // Return info about workers_
  for (int i = 0; i < workers_.size(); ++i) {
    WorkerField* worker = reply->add_worker();
    worker->set_alive(workers_[i].alive);
    worker->set_objstoreid(workers_[i].objstoreid);
  }
  // Return info about objstores_
  for (int i = 0; i < objstores_.size(); ++i) {
    ObjStoreField* objstore = reply->add_objstore();
    objstore->set_alive(objstores_[i].alive);
    objstore->set_address(objstores_[i].address);
  }
  // Return info about reference_counts_
  for (int i = 0; i < reference_counts_.size(); ++i) {
    reply->add_reference_count(reference_counts_[i]);
  }
  // Return info about reverse_target_objrefs_
  for (int i = 0; i < reverse_target_objrefs_.size(); ++i) {
    ObjRefList* reverse_target_objrefs = reply->add_reverse_target_objrefs();
    for (int j = 0; j < reverse_target_objrefs_[i].size(); ++j) {
      reverse_target_objrefs->add_objref(reverse_target_objrefs_[i][j]);
    }
  }
  // Return info about alias_notification_queue_
  for (int i = 0; i < alias_notification_queue_.size(); ++i) {
    AliasNotification* alias_notification = reply->add_alias_notification();
    alias_notification->set_objstoreid(alias_notification_queue_[i].first);
    alias_notification->set_alias_objref(alias_notification_queue_[i].second.first);
    alias_notification->set_canonical_objref(alias_notification_queue_[i].second.second);
  }
  // Return info about contained_objrefs_
  for (int i = 0; i < contained_objrefs_.size(); ++i) {
    ObjRefList* contained_objrefs = reply->add_contained_objrefs();
    for (int j = 0; j < contained_objrefs_[i].size(); ++j) {
      contained_objrefs->add_objref(contained_objrefs_[i][j]);
    }
  }
  // Return info about pull_queue_
  for (int i = 0; i < pull_queue_.size(); ++i) {
    Pull* pull = reply->add_pull();
    pull->set_workerid(pull_queue_[i].first);
    pull->set_objref(pull_queue_[i].second);
  }
  // Return info about objtable_
  for (int i = 0; i < objtable_.size(); ++i) {
    ObjStoreList* location_list = reply->add_location_list();
    for (int j = 0; j < objtable_[i].size(); ++j) {
      location_list->add_objstoreid(objtable_[i][j]);
    }
  }
  // Return info about current_tasks_
  for (int i = 0; i < current_tasks_.size(); ++i) {
    CurrentTask* current_task = reply->add_current_task();
    current_task->set_taskid(current_tasks_[i].first);
    current_task->set_new_task(current_tasks_[i].second.first);
    current_task->set_num_spawned(current_tasks_[i].second.second);
  }
  // Return info about target_objrefs_
  for (int i = 0; i < target_objrefs_.size(); ++i) {
    reply->add_target_objref(target_objrefs_[i]);
  }
  // Return info about fntable_
  auto function_table = reply->mutable_function_table();
  for (const auto& entry : fntable_) {
    (*function_table)[entry.first].set_num_return_vals(entry.second.num_return_vals());
    for (const WorkerId& worker : entry.second.workers()) {
      (*function_table)[entry.first].add_workerid(worker);
    }
  }
  // Return info about task_queue_
  for (const auto& entry : task_queue_) {
    reply->add_taskid(entry.first);
  }
  // Return info about avail_workers_
  for (const WorkerId& entry : avail_workers_) {
    reply->add_avail_worker(entry);
  }
  // Return info about computation_graph_.tasks_
  for (int i = 0; i < computation_graph_.num_tasks(); ++i) {
    Call* task = reply->add_task();
    task->CopyFrom(computation_graph_.get_task(i));
  }
  // Return info about computation_graph_.spawned_tasks_
  for (int i = 0; i < computation_graph_.num_tasks(); ++i) {
    TaskList* spawned = reply->add_spawned_task();
    std::vector<TaskId> spawned_tasks = computation_graph_.get_spawned_tasks(i);
    for (int j = 0; j < spawned_tasks.size(); ++j) {
      spawned->add_taskid(spawned_tasks[j]);
    }
  }
  release_all_locks();
}

// pick_objstore assumes that objtable_lock_ has been acquired
// pick_objstore must be called with a canonical_objref
ObjStoreId SchedulerService::pick_objstore(ObjRef canonical_objref) {
  std::mt19937 rng;
  if (!is_canonical(canonical_objref)) {
    ORCH_LOG(ORCH_FATAL, "Attempting to call pick_objstore with a non-canonical objref, (objref " << canonical_objref << ")");
  }
  std::uniform_int_distribution<int> uni(0, objtable_[canonical_objref].size() - 1);
  ObjStoreId objstoreid = objtable_[canonical_objref][uni(rng)];
  return objstoreid;
}

bool SchedulerService::is_canonical(ObjRef objref) {
  std::lock_guard<std::mutex> lock(target_objrefs_lock_);
  if (target_objrefs_[objref] == UNITIALIZED_ALIAS) {
    ORCH_LOG(ORCH_FATAL, "Attempting to call is_canonical on an objref for which aliasing is not complete or the object is not ready, target_objrefs_[objref] == UNITIALIZED_ALIAS for objref " << objref << ".");
  }
  return objref == target_objrefs_[objref];
}

// This returns true if the objref is a canonical objref which has already been
// stored in an object store or if it is a non-canonical objref which has
// already been aliased (it's corresponding canonical objref doesn't have to be
// present in an object store yet). This returns false otherwise.
bool SchedulerService::already_present(ObjRef objref) {
  target_objrefs_lock_.lock();
  ObjRef target_objref = target_objrefs_[objref];
  target_objrefs_lock_.unlock();
  if (target_objref == UNITIALIZED_ALIAS) {
    return false;
  }
  if (target_objref != objref) {
    return true;
  }
  objtable_lock_.lock();
  bool is_present = (objtable_[objref].size() > 0);
  objtable_lock_.unlock();
  return is_present;
}

void SchedulerService::perform_pulls() {
  std::lock_guard<std::mutex> pull_queue_lock(pull_queue_lock_);
  // Complete all pull tasks that can be completed.
  for (int i = 0; i < pull_queue_.size(); ++i) {
    const std::pair<WorkerId, ObjRef>& pull = pull_queue_[i];
    ObjRef objref = pull.second;
    WorkerId workerid = pull.first;
    if (!has_canonical_objref(objref)) {
      ORCH_LOG(ORCH_ALIAS, "objref " << objref << " does not have a canonical_objref, so continuing");
      continue;
    }
    ObjRef canonical_objref = get_canonical_objref(objref);
    ORCH_LOG(ORCH_DEBUG, "attempting to pull objref " << pull.second << " with canonical objref " << canonical_objref << " to objstore " << get_store(workerid));

    objtable_lock_.lock();
    int num_stores = objtable_[canonical_objref].size();
    objtable_lock_.unlock();

    if (num_stores > 0) {
      bool delivery_started_successfully = true;
      {
        std::lock_guard<std::mutex> objtable_lock(objtable_lock_);
        if (!std::binary_search(objtable_[canonical_objref].begin(), objtable_[canonical_objref].end(), get_store(workerid))) {
          // The worker's local object store does not already contain objref, so ship
          // it there from an object store that does have it.
          ObjStoreId objstoreid = pick_objstore(canonical_objref);
          delivery_started_successfully = deliver_object(canonical_objref, objstoreid, get_store(workerid));
        }
      }
      if (delivery_started_successfully) {
        {
          // Notify the relevant objstore about potential aliasing when it's ready
          std::lock_guard<std::mutex> alias_notification_queue_lock(alias_notification_queue_lock_);
          alias_notification_queue_.push_back(std::make_pair(get_store(workerid), std::make_pair(objref, canonical_objref)));
        }
        // Remove the pull task from the queue
        std::swap(pull_queue_[i], pull_queue_[pull_queue_.size() - 1]);
        pull_queue_.pop_back();
        i -= 1;
      }
    }
  }
}

void SchedulerService::schedule_tasks_naively() {
  // ORCH_LOG(ORCH_DEBUG, "In schedule_tasks: acquiring computation_graph");
  std::lock_guard<std::mutex> computation_graph_lock(computation_graph_lock_); // TODO(rkn): Ideally we wouldn't hold these locks the whole time, it can probably be more fine-grained
  // ORCH_LOG(ORCH_DEBUG, "In schedule_tasks: acquiring fntable");
  std::lock_guard<std::mutex> fntable_lock(fntable_lock_);
  // ORCH_LOG(ORCH_DEBUG, "In schedule_tasks: acquiring avail_workers");
  std::lock_guard<std::mutex> avail_workers_lock(avail_workers_lock_);
  // ORCH_LOG(ORCH_DEBUG, "In schedule_tasks: acquiring task_queue");
  std::lock_guard<std::mutex> task_queue_lock(task_queue_lock_);
  for (int i = 0; i < avail_workers_.size(); ++i) {
    // Submit all tasks whose arguments are ready.
    // ORCH_LOG(ORCH_DEBUG, "In schedule_tasks: AAAAA");
    WorkerId workerid = avail_workers_[i];
    for (auto it = task_queue_.begin(); it != task_queue_.end(); ++it) {
      // ORCH_LOG(ORCH_DEBUG, "In schedule_tasks: BBBBB");
      // The use of erase(it) below invalidates the iterator, but we
      // immediately break out of the inner loop, so the iterator is not used
      // after the erase
      std::pair<TaskId, bool> task_info = *it;
      TaskId taskid = task_info.first;
      bool is_new_task = task_info.second;
      // ORCH_LOG(ORCH_DEBUG, "In schedule_tasks: CCCCC");
      const Call& task = computation_graph_.get_task(taskid);
      auto& workers = fntable_[task.name()].workers();
      // ORCH_LOG(ORCH_DEBUG, "In schedule_tasks: DDDDD");
      if (std::binary_search(workers.begin(), workers.end(), workerid) && can_run(task)) {
        // ORCH_LOG(ORCH_DEBUG, "In schedule_tasks: EEEEE");
        // submit_task(taskid, is_new_task, workerid);

        if (!workers_[workerid].alive) {
          // TODO(rkn): Remove this check, this is just for debugging.
          ORCH_LOG(ORCH_FATAL, "At line 798: Scheduler about to submit_task to worker " << workerid << ", but this worker is no longer alive.");
        }

        // TODO(rkn): submit_task updates current_tasks_, and avail_workers_ is
        // updated below. But really these should be updated in the same place.
        if (submit_task(taskid, task, is_new_task, workerid)) {
          // If the task is successfully submitted, remove it from task_queue_.
          // ORCH_LOG(ORCH_DEBUG, "In schedule_tasks: FFFFF");
          task_queue_.erase(it);
          // ORCH_LOG(ORCH_DEBUG, "In schedule_tasks: GGGGG");
          std::swap(avail_workers_[i], avail_workers_[avail_workers_.size() - 1]);
          // ORCH_LOG(ORCH_DEBUG, "In schedule_tasks: HHHHH");
          avail_workers_.pop_back();
          // ORCH_LOG(ORCH_DEBUG, "In schedule_tasks: IIIII");
          i -= 1;
          break;
        }
      }
      // ORCH_LOG(ORCH_DEBUG, "In schedule_tasks: JJJJJ");
    }
    // ORCH_LOG(ORCH_DEBUG, "In schedule_tasks: KKKKK");
  }
  // ORCH_LOG(ORCH_DEBUG, "In schedule_tasks: releasing fntable_lock");
}

void SchedulerService::schedule_tasks_location_aware() {
  std::lock_guard<std::mutex> fntable_lock(fntable_lock_);
  std::lock_guard<std::mutex> avail_workers_lock(avail_workers_lock_);
  std::lock_guard<std::mutex> task_queue_lock(task_queue_lock_);
  for (int i = 0; i < avail_workers_.size(); ++i) {
    // Submit all tasks whose arguments are ready.
    WorkerId workerid = avail_workers_[i];
    ObjStoreId objstoreid = workers_[workerid].objstoreid;
    auto bestit = task_queue_.end(); // keep track of the task that fits the worker best so far
    size_t min_num_shipped_objects = std::numeric_limits<size_t>::max(); // number of objects that need to be transfered for this worker
    for (auto it = task_queue_.begin(); it != task_queue_.end(); ++it) {
      const Call& task = *(*it);
      auto& workers = fntable_[task.name()].workers();
      if (std::binary_search(workers.begin(), workers.end(), workerid) && can_run(task)) {
        // determine how many objects would need to be shipped
        size_t num_shipped_objects = 0;
        for (int j = 0; j < task.arg_size(); ++j) {
          if (!task.arg(j).has_obj()) {
            ObjRef objref = task.arg(j).ref();
            if (!has_canonical_objref(objref)) {
              ORCH_LOG(ORCH_FATAL, "no canonical object ref found even though task is ready; that should not be possible!");
            }
            ObjRef canonical_objref = get_canonical_objref(objref);
            // check if the object is already in the local object store
            if (!std::binary_search(objtable_[canonical_objref].begin(), objtable_[canonical_objref].end(), objstoreid)) {
              num_shipped_objects += 1;
            }
          }
        }
        if (num_shipped_objects < min_num_shipped_objects) {
          min_num_shipped_objects = num_shipped_objects;
          bestit = it;
        }
      }
    }
    // if we found a suitable task
    if (bestit != task_queue_.end()) {
      submit_task(std::move(*bestit), workerid);
      task_queue_.erase(bestit);
      std::swap(avail_workers_[i], avail_workers_[avail_workers_.size() - 1]);
      avail_workers_.pop_back();
      i -= 1;
    }
  }
}

void SchedulerService::perform_notify_aliases() {
  ORCH_LOG(ORCH_DEBUG, "In perform_notify_aliases: Attempting to acquire alias_notification_queue_lock!!");
  std::lock_guard<std::mutex> alias_notification_queue_lock(alias_notification_queue_lock_);
  for (int i = 0; i < alias_notification_queue_.size(); ++i) {
    const std::pair<WorkerId, std::pair<ObjRef, ObjRef> > alias_notification = alias_notification_queue_[i];
    ObjStoreId objstoreid = alias_notification.first;
    ObjRef alias_objref = alias_notification.second.first;
    ObjRef canonical_objref = alias_notification.second.second;
    if (attempt_notify_alias(objstoreid, alias_objref, canonical_objref)) { // this locks both the objstore_ and objtable_
      // the attempt to notify the objstore of the objref aliasing succeeded, so remove the notification task from the queue
      std::swap(alias_notification_queue_[i], alias_notification_queue_[alias_notification_queue_.size() - 1]);
      alias_notification_queue_.pop_back();
      i -= 1;
    }
  }
  ORCH_LOG(ORCH_DEBUG, "Exiting perform_notify_aliases");
}

bool SchedulerService::has_canonical_objref(ObjRef objref) {
  std::lock_guard<std::mutex> lock(target_objrefs_lock_);
  ObjRef objref_temp = objref;
  while (true) {
    if (objref_temp >= target_objrefs_.size()) {
      ORCH_LOG(ORCH_FATAL, "Attempting to index target_objrefs_ with objref " << objref_temp << ", but target_objrefs_.size() = " << target_objrefs_.size());
    }
    if (target_objrefs_[objref_temp] == UNITIALIZED_ALIAS) {
      return false;
    }
    if (target_objrefs_[objref_temp] == objref_temp) {
      return true;
    }
    objref_temp = target_objrefs_[objref_temp];
  }
}

ObjRef SchedulerService::get_canonical_objref(ObjRef objref) {
  // get_canonical_objref assumes that has_canonical_objref(objref) is true
  std::lock_guard<std::mutex> lock(target_objrefs_lock_);
  ObjRef objref_temp = objref;
  while (true) {
    if (objref_temp >= target_objrefs_.size()) {
      ORCH_LOG(ORCH_FATAL, "Attempting to index target_objrefs_ with objref " << objref_temp << ", but target_objrefs_.size() = " << target_objrefs_.size());
    }
    if (target_objrefs_[objref_temp] == UNITIALIZED_ALIAS) {
      ORCH_LOG(ORCH_FATAL, "Attempting to get canonical objref for objref " << objref << ", which aliases, objref " << objref_temp << ", but target_objrefs_[objref_temp] == UNITIALIZED_ALIAS for objref_temp = " << objref_temp << ".");
    }
    if (target_objrefs_[objref_temp] == objref_temp) {
      return objref_temp;
    }
    objref_temp = target_objrefs_[objref_temp];
    ORCH_LOG(ORCH_ALIAS, "Looping in get_canonical_objref.");
  }
}

bool SchedulerService::attempt_notify_alias(ObjStoreId objstoreid, ObjRef alias_objref, ObjRef canonical_objref) {
  // return true if successful and false otherwise
  if (alias_objref == canonical_objref) {
    // no need to do anything
    return true;
  }
  {
    std::lock_guard<std::mutex> lock(objtable_lock_);
    if (!std::binary_search(objtable_[canonical_objref].begin(), objtable_[canonical_objref].end(), objstoreid)) {
      // the objstore doesn't have the object for canonical_objref yet, so it's too early to notify the objstore about the alias
      return false;
    }
  }
  ClientContext context;
  AckReply reply;
  NotifyAliasRequest request;
  request.set_alias_objref(alias_objref);
  request.set_canonical_objref(canonical_objref);
  objstores_lock_.lock();
  objstores_[objstoreid].objstore_stub->NotifyAlias(&context, request, &reply);
  objstores_lock_.unlock();
  return true;
}

void SchedulerService::deallocate_object(ObjRef canonical_objref) {
  // deallocate_object should only be called from decrement_ref_count (note that
  // deallocate_object also recursively calls decrement_ref_count). Both of
  // these methods require reference_counts_lock_ to have been acquired, and
  // so the lock must before outside of these methods (it is acquired in
  // DecrementRefCount).
  ORCH_LOG(ORCH_DEBUG, "DEALLOCATING CANONICAL_OBJREF " << canonical_objref << ".");
  ORCH_LOG(ORCH_REFCOUNT, "Deallocating canonical_objref " << canonical_objref << ".");
  {
    std::lock_guard<std::mutex> objtable_lock(objtable_lock_);
    auto &objstores = objtable_[canonical_objref];
    std::lock_guard<std::mutex> objstores_lock(objstores_lock_); // TODO(rkn): Should this be inside the for loop instead?
    for (int i = 0; i < objstores.size(); ++i) {
      ClientContext context;
      AckReply reply;
      DeallocateObjectRequest request;
      request.set_canonical_objref(canonical_objref);
      ObjStoreId objstoreid = objstores[i];
      ORCH_LOG(ORCH_REFCOUNT, "Attempting to deallocate canonical_objref " << canonical_objref << " from objstore " << objstoreid);
      objstores_[objstoreid].objstore_stub->DeallocateObject(&context, request, &reply);
    }
  }
  decrement_ref_count(contained_objrefs_[canonical_objref]);
}

void SchedulerService::increment_ref_count(std::vector<ObjRef> &objrefs) {
  // increment_ref_count assumes that reference_counts_lock_ has been acquired already
  for (int i = 0; i < objrefs.size(); ++i) {
    ObjRef objref = objrefs[i];
    if (reference_counts_[objref] == DEALLOCATED) {
      ORCH_LOG(ORCH_FATAL, "Attempting to increment the reference count for objref " << objref << ", but this object appears to have been deallocated already.");
    }
    reference_counts_[objref] += 1;
    ORCH_LOG(ORCH_REFCOUNT, "Incremented ref count for objref " << objref <<". New reference count is " << reference_counts_[objref]);
  }
}

void SchedulerService::decrement_ref_count(std::vector<ObjRef> &objrefs) {
  // decrement_ref_count assumes that reference_counts_lock_ has been acquired already
  for (int i = 0; i < objrefs.size(); ++i) {
    ObjRef objref = objrefs[i];
    if (reference_counts_[objref] == DEALLOCATED) {
      ORCH_LOG(ORCH_FATAL, "Attempting to decrement the reference count for objref " << objref << ", but this object appears to have been deallocated already.");
    }
    if (reference_counts_[objref] == 0) {
      ORCH_LOG(ORCH_FATAL, "Attempting to decrement the reference count for objref " << objref << ", but the reference count for this object is already 0.");
    }
    reference_counts_[objref] -= 1;
    ORCH_LOG(ORCH_REFCOUNT, "Decremented ref count for objref " << objref << ". New reference count is " << reference_counts_[objref]);
    // See if we can deallocate the object
    std::vector<ObjRef> equivalent_objrefs;
    get_equivalent_objrefs(objref, equivalent_objrefs);
    bool can_deallocate = true;
    for (int j = 0; j < equivalent_objrefs.size(); ++j) {
      if (reference_counts_[equivalent_objrefs[j]] != 0) {
        can_deallocate = false;
        break;
      }
    }
    if (can_deallocate) {
      ObjRef canonical_objref = equivalent_objrefs[0];
      if (!is_canonical(canonical_objref)) {
        ORCH_LOG(ORCH_FATAL, "canonical_objref is not canonical.");
      }
      deallocate_object(canonical_objref);
      for (int j = 0; j < equivalent_objrefs.size(); ++j) {
        reference_counts_[equivalent_objrefs[j]] = DEALLOCATED;
      }
    }
  }
}

void SchedulerService::upstream_objrefs(ObjRef objref, std::vector<ObjRef> &objrefs) {
  // upstream_objrefs assumes that the lock reverse_target_objrefs_lock_ has been acquired
  objrefs.push_back(objref);
  for (int i = 0; i < reverse_target_objrefs_[objref].size(); ++i) {
    upstream_objrefs(reverse_target_objrefs_[objref][i], objrefs);
  }
}

void SchedulerService::get_equivalent_objrefs(ObjRef objref, std::vector<ObjRef> &equivalent_objrefs) {
  std::lock_guard<std::mutex> target_objrefs_lock(target_objrefs_lock_);
  ObjRef downstream_objref = objref;
  while (target_objrefs_[downstream_objref] != downstream_objref && target_objrefs_[downstream_objref] != UNITIALIZED_ALIAS) {
    ORCH_LOG(ORCH_ALIAS, "Looping in get_equivalent_objrefs");
    downstream_objref = target_objrefs_[downstream_objref];
  }
  std::lock_guard<std::mutex> reverse_target_objrefs_lock(reverse_target_objrefs_lock_);
  upstream_objrefs(downstream_objref, equivalent_objrefs);
}

void start_scheduler_service(const char* service_addr, SchedulingAlgorithmType scheduling_algorithm) {
  std::string service_address(service_addr);
  std::string::iterator split_point = split_ip_address(service_address);
  std::string port;
  port.assign(split_point, service_address.end());
  SchedulerService service(scheduling_algorithm);
  ServerBuilder builder;
  builder.AddListeningPort(std::string("0.0.0.0:") + port, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<Server> server(builder.BuildAndStart());
  server->Wait();
}

char* get_cmd_option(char** begin, char** end, const std::string& option) {
  char** it = std::find(begin, end, option);
  if (it != end && ++it != end) {
    return *it;
  }
  return 0;
}

void SchedulerService::recover_from_failed_objstore(ObjStoreId objstoreid) {
  // This method must do the following:
  //   - remove objstoreid from all relevant entries in objtable_
  //   - reassign all of this object store's workers to a different object store TODO(rkn): Or should we just assume those workers are all dead.
  // ORCH_LOG(ORCH_DEBUG, "SCHEDULER IN RECOVER_FROM_FAILED_OBJSTORE: 11111");
  // TODO(rkn): Right now we are blocking here presumably because some scheduler methods are waiting for a killed worker and holding a lock.
  // TRY DELETING THE RELEVANT OBJSTORE AND WORKER SMART POINTERS
  // objstores_[objstoreid].objstore_stub.reset(); // TODO(rkn): Really I'd prefer to do this inside a lock
  acquire_all_locks();
  // ORCH_LOG(ORCH_DEBUG, "SCHEDULER IN RECOVER_FROM_FAILED_OBJSTORE: 22222");
  // Update fntable_
  for (auto it = fntable_.begin(); it != fntable_.end(); ++it) {
    auto workers = it->second.workers();
    auto location = std::find(workers.begin(), workers.end(), objstoreid);
    if (location != workers.end()) {
      workers.erase(location);
    }
  }
  // Update objtable_
  // REDO THIS AS FOLLOWS:
  // 1) make a list of all the objects that are no longer in the object store (either lost or were never present)
  // 2) remove all of the objects that will be created by a current task
  // 3) remove all of the objects that will be created by something in the task queue

  std::vector<std::vector<ObjRef> > tasks_to_rerun(objtable_.size()); // if tasks_to_rerun[i] is non-empty, then we must rerun task i to recompute the objrefs in tasks_to_rerun[i]
  std::vector<ObjRef> objrefs_to_increment;
  for (ObjRef i = 0; i < objtable_.size(); ++i) {
    auto it = std::find(objtable_[i].begin(), objtable_[i].end(), objstoreid);
    if (it != objtable_[i].end()) {
      if (objtable_[i].size() == 1) {
        ORCH_LOG(ORCH_DEBUG, "LOST OBJREF " << i << ".........");
      }
      objtable_[i].erase(it);
      if (objtable_[i].size() == 0) {
        // We lost object i, so note that we need to recompute it
        TaskId taskid = computation_graph_.get_creator_taskid(i);
        ORCH_LOG(ORCH_DEBUG, "We lost objref " << i << " so we need to re-run task " << taskid << " to recompute it.");
        tasks_to_rerun[taskid].push_back(i);
        contained_objrefs_[i].clear(); // When we reconstruct the object, we will call AddContainedObjRefs again, which will fail if it is not cleared now.
        const Call& task = computation_graph_.get_task(taskid); // TODO(rkn): Check this code, and factor it out with the duplicate below
        for (int j = 0; j < task.result_size(); ++j) {
          objrefs_to_increment.push_back(task.result(j));
        }
      }
    } else if (objtable_[i].size() == 0) {
      // TODO(rkn): Document this better. This is to handle the situation where
      // a worker finishes a task and has called WorkerReady, but the object
      // store that was storing the output of that task dies before it calls
      // ObjReady. For each object that isn't in any object store, if it isn't
      // going to be created by a currently running task or by a task in the
      // task_queue_, then we must rerun the creator task.
      bool must_rerun_task = true;
      TaskId taskid = computation_graph_.get_creator_taskid(i);
      for (int j = 0; j < task_queue_.size(); ++j) {
        if (task_queue_[j].first == taskid) {
          must_rerun_task = false;
        }
      }
      for (int j = 0; j < current_tasks_.size(); ++j) {
        if (workers_[j].alive && current_tasks_[j].first == taskid) {
          must_rerun_task = false;
        }
      }
      if (must_rerun_task) {
        ORCH_LOG(ORCH_DEBUG, "objref " << i << " is not present and will not be created by any currently running task or any task in the task_queue_, so we must rerun the task. This means that the worker finished but the object store died before calling ObjReady.");
        tasks_to_rerun[taskid].push_back(i);
        contained_objrefs_[i].clear(); // When we reconstruct the object, we will call AddContainedObjRefs again, which will fail if it is not cleared now.
        const Call& task = computation_graph_.get_task(taskid); // TODO(rkn): Check this code, and factor it out with the duplicate above
        for (int j = 0; j < task.result_size(); ++j) {
          objrefs_to_increment.push_back(task.result(j));
        }
      }
    }
  }
  // We are pushing tasks onto the task_queue_. When these tasks get
  // deserialized, the objrefs for the results will be decremented. Normally,
  // the corresponding increment happens in RemoteCall, but that isn't happening
  // when we handle fault tolerance. So we do it manually here and keep track of
  // which objrefs to increment in objref_to_increment. TODO(rkn): figure out a
  // better way to do this.
  increment_ref_count(objrefs_to_increment);
  // Run the tasks.
  for (int i = 0; i < tasks_to_rerun.size(); ++i) {
    if (tasks_to_rerun[i].size() > 0) {
      // We lost some objects that were created by this task, so rerun the task.
      // task_queue_.push_back(std::make_pair(i, TaskRunType::RECOVER));
      task_queue_.push_back(std::make_pair(i, false)); // false indicates that this task is being rerun
    }
  }

  // ORCH_LOG(ORCH_DEBUG, "SCHEDULER IN RECOVER_FROM_FAILED_OBJSTORE: AAAAA");
  // Update workers_
  for (int i = 0; i < workers_.size(); ++i) {
    if (workers_[i].objstoreid == objstoreid) { // TODO(rkn): Really I'd prefer to do this inside a lock
      ORCH_LOG(ORCH_DEBUG, "SCHEDULER IN RECOVER_FROM_FAILED_OBJSTORE: attempting to kill worker " << i);
      kill_worker(i); // If an object store dies, assume that all workers connected to that object store also died (and kill them just to be safe).
    }
  }
  // ORCH_LOG(ORCH_DEBUG, "SCHEDULER IN RECOVER_FROM_FAILED_OBJSTORE: BBBBB");
  // Update avail_workers_
  ORCH_LOG(ORCH_DEBUG, "BEFORE RECOVERY, THE AVAILABLE WORKERS ARE:")
  for (int i = 0; i < avail_workers_.size(); ++i) {
    ORCH_LOG(ORCH_DEBUG, "-----" << avail_workers_[i]);
  }
  for (int i = 0; i < avail_workers_.size(); ++i) {
    if (!workers_[avail_workers_[i]].alive) {
      // avail_workers_[i] is no longer alive, so remove it from the avail_workers_
      std::swap(avail_workers_[i], avail_workers_[avail_workers_.size() - 1]);
      avail_workers_.pop_back();
      --i;
    }
  }
  ORCH_LOG(ORCH_DEBUG, "AFTER RECOVERY, THE AVAILABLE WORKERS ARE:")
  for (int i = 0; i < avail_workers_.size(); ++i) {
    ORCH_LOG(ORCH_DEBUG, "-----" << avail_workers_[i]);
  }
  // Update current_tasks_
  for (int i = 0; i < current_tasks_.size(); ++i) {
    if (workers_[i].objstoreid == objstoreid) {
      TaskId taskid = current_tasks_[i].first;
      if (taskid != NO_TASK) {
        ORCH_LOG(ORCH_DEBUG, "Adding task " << taskid << " to the task_queue_ because it was in the middle of running during the recovery.");
        // task_queue_.push_back(std::make_pair(taskid, TaskRunType::RETRY));
        task_queue_.push_back(std::make_pair(taskid, false)); // false indicates that this task is being rerun
      }
    }
  }
  // Update reference counts_
  // TODO(rkn): Implement this. If we don't do anything, the reference counts should probably be an overestimate.
  // Update pull_queue_
  // TODO(rkn): Implement this. Any pull going to a dead worker should be invalidated.
  // Update alias_notification_queue_
  // TODO(rkn): Implement this. Any alias notification going to a dead objstore should be invalidated.

  // Recover lost objects
  // for (tasks that need to be rerun) {
  //   rerun task with the objects that need to be computed
  // }

  release_all_locks();
  // ORCH_LOG(ORCH_DEBUG, "SCHEDULER IN RECOVER_FROM_FAILED_OBJSTORE: 33333");
  schedule();
}

void SchedulerService::recover_from_failed_worker(WorkerId workerid) {
  // This method must do the following:
  //   - remove the worker that died from all relevant entries in fntable_
  //   - remove the worker from avail_workers_
  //   - remove the worker from current_tasks_ TODO(rkn): or not?

}

// This method assumes that objstores_lock_ has been acquired.
void SchedulerService::kill_objstore(ObjStoreId objstoreid) {
  ClientContext terminate_context;
  AckReply terminate_reply;
  TerminateObjStoreRequest terminate_request;
  objstores_[objstoreid].objstore_stub->Terminate(&terminate_context, terminate_request, &terminate_reply);
  objstores_[objstoreid].alive = false;
}

// This method assumes that workers_lock_ has been acquired.
void SchedulerService::kill_worker(WorkerId workerid) {
  ClientContext terminate_context;
  AckReply terminate_reply;
  TerminateWorkerRequest terminate_request;
  workers_[workerid].worker_stub->Terminate(&terminate_context, terminate_request, &terminate_reply);
  workers_[workerid].alive = false;
  workers_[workerid].worker_stub.reset();
}

// This method defines the canonical order in which locks should be acquired.
void SchedulerService::acquire_all_locks() {
  // TODO(rkn): alias_notification_queue_lock_ may need to come before objtable_lock_
  // ORCH_LOG(ORCH_DEBUG, "In ACQUIRE_ALL_LOCKS: acquiring computation_graph");
  computation_graph_lock_.lock();
  // ORCH_LOG(ORCH_DEBUG, "In ACQUIRE_ALL_LOCKS: acquiring current_tasks");
  current_tasks_lock_.lock();
  // ORCH_LOG(ORCH_DEBUG, "In ACQUIRE_ALL_LOCKS: acquiring reference_counts");
  reference_counts_lock_.lock();
  // ORCH_LOG(ORCH_DEBUG, "In ACQUIRE_ALL_LOCKS: acquiring contained_objrefs");
  contained_objrefs_lock_.lock();
  // ORCH_LOG(ORCH_DEBUG, "In ACQUIRE_ALL_LOCKS: acquiring objtable");
  objtable_lock_.lock();
  // ORCH_LOG(ORCH_DEBUG, "In ACQUIRE_ALL_LOCKS: acquiring pull_queue");
  pull_queue_lock_.lock();
  // ORCH_LOG(ORCH_DEBUG, "In ACQUIRE_ALL_LOCKS: acquiring target_objrefs");
  target_objrefs_lock_.lock();
  // ORCH_LOG(ORCH_DEBUG, "In ACQUIRE_ALL_LOCKS: acquiring reverse_target_objrefs");
  reverse_target_objrefs_lock_.lock();
  // ORCH_LOG(ORCH_DEBUG, "In ACQUIRE_ALL_LOCKS: acquiring fntable");
  fntable_lock_.lock();
  // ORCH_LOG(ORCH_DEBUG, "In ACQUIRE_ALL_LOCKS: acquiring avail_workers");
  avail_workers_lock_.lock();
  // ORCH_LOG(ORCH_DEBUG, "In ACQUIRE_ALL_LOCKS: acquiring task_queue");
  task_queue_lock_.lock();
  // ORCH_LOG(ORCH_DEBUG, "In ACQUIRE_ALL_LOCKS: acquiring alias_notification_queue");
  alias_notification_queue_lock_.lock();
  // ORCH_LOG(ORCH_DEBUG, "In ACQUIRE_ALL_LOCKS: acquiring objstores");
  objstores_lock_.lock();
  // ORCH_LOG(ORCH_DEBUG, "In ACQUIRE_ALL_LOCKS: acquiring workers");
  workers_lock_.lock();
}

void SchedulerService::release_all_locks() {
  // These locks should appear in the same order as in acquire_all_locks().
  // ORCH_LOG(ORCH_DEBUG, "RELEASING ALL LOCKS");
  computation_graph_lock_.unlock();
  current_tasks_lock_.unlock();
  reference_counts_lock_.unlock();
  contained_objrefs_lock_.unlock();
  objtable_lock_.unlock();
  pull_queue_lock_.unlock();
  target_objrefs_lock_.unlock();
  reverse_target_objrefs_lock_.unlock();
  fntable_lock_.unlock();
  avail_workers_lock_.unlock();
  task_queue_lock_.unlock();
  alias_notification_queue_lock_.unlock();
  objstores_lock_.unlock();
  workers_lock_.unlock();
}

int main(int argc, char** argv) {
  SchedulingAlgorithmType scheduling_algorithm = SCHEDULING_ALGORITHM_LOCALITY_AWARE;
  if (argc < 2) {
    ORCH_LOG(ORCH_FATAL, "scheduler: expected at least one argument (scheduler ip address)");
    return 1;
  }
  if (argc > 2) {
    char* scheduling_algorithm_name = get_cmd_option(argv, argv + argc, "--scheduler-algorithm");
    if (scheduling_algorithm_name) {
      if(std::string(scheduling_algorithm_name) == "naive") {
        std::cout << "using 'naive' scheduler" << std::endl;
        scheduling_algorithm = SCHEDULING_ALGORITHM_NAIVE;
      }
      if(std::string(scheduling_algorithm_name) == "locality_aware") {
        std::cout << "using 'locality aware' scheduler" << std::endl;
        scheduling_algorithm = SCHEDULING_ALGORITHM_LOCALITY_AWARE;
      }
    }
  }
  start_scheduler_service(argv[1], scheduling_algorithm);
  return 0;
}
