#ifndef ORCHESTRA_SCHEDULER_H
#define ORCHESTRA_SCHEDULER_H


#include <deque>
#include <memory>
#include <algorithm>
#include <iostream>
#include <limits>

#include <grpc++/grpc++.h>

#include "orchestra/orchestra.h"
#include "orchestra.grpc.pb.h"
#include "types.pb.h"

#include "computationgraph.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerReader;
using grpc::ServerContext;
using grpc::Status;

using grpc::ClientContext;

using grpc::Channel;

typedef size_t RefCount;

const ObjRef UNITIALIZED_ALIAS = std::numeric_limits<ObjRef>::max();
const RefCount DEALLOCATED = std::numeric_limits<RefCount>::max();

// The following type is used to mark the state of a task that has been
// submitted to the task_queue_ or is in current_tasks_.
// NEW: This indicates that the task is being put in the task_queue_ or running
// as a current task for the first time.
// RETRY: This indicates that the task has been submitted to a worker before,
// but the worker died before the task completed.
// RECOVER: This indicates that the task has been run before and finished, but
// due to a failure, we lost an object that was created by this task and so we
// are rerunning it. The most important part of the recover
// enum TaskRunType {NEW = 0, RETRY = 1, RECOVER = 2};

struct WorkerHandle {
  bool alive; // true if the worker is alive, false if the worker is dead
  std::shared_ptr<Channel> channel;
  std::unique_ptr<WorkerService::Stub> worker_stub;
  ObjStoreId objstoreid;
};

struct ObjStoreHandle {
  bool alive; // true if the objstore is alive, false if the objstore is dead
  std::shared_ptr<Channel> channel;
  std::unique_ptr<ObjStore::Stub> objstore_stub;
  std::string address;
};

enum SchedulingAlgorithmType {
  SCHEDULING_ALGORITHM_NAIVE = 0,
  SCHEDULING_ALGORITHM_LOCALITY_AWARE = 1
};

class SchedulerService : public Scheduler::Service {
public:
  SchedulerService(SchedulingAlgorithmType scheduling_algorithm);

  Status RemoteCall(ServerContext* context, const RemoteCallRequest* request, RemoteCallReply* reply) override;
  Status PushObj(ServerContext* context, const PushObjRequest* request, PushObjReply* reply) override;
  Status RequestObj(ServerContext* context, const RequestObjRequest* request, AckReply* reply) override;
  Status AliasObjRefs(ServerContext* context, const AliasObjRefsRequest* request, AckReply* reply) override;
  Status RegisterObjStore(ServerContext* context, const RegisterObjStoreRequest* request, RegisterObjStoreReply* reply) override;
  Status RegisterWorker(ServerContext* context, const RegisterWorkerRequest* request, RegisterWorkerReply* reply) override;
  Status RegisterFunction(ServerContext* context, const RegisterFunctionRequest* request, AckReply* reply) override;
  Status ObjReady(ServerContext* context, const ObjReadyRequest* request, AckReply* reply) override;
  Status WorkerReady(ServerContext* context, const WorkerReadyRequest* request, AckReply* reply) override;
  Status IncrementRefCount(ServerContext* context, const IncrementRefCountRequest* request, AckReply* reply) override;
  Status DecrementRefCount(ServerContext* context, const DecrementRefCountRequest* request, AckReply* reply) override;
  Status AddContainedObjRefs(ServerContext* context, const AddContainedObjRefsRequest* request, AckReply* reply) override;
  Status SchedulerInfo(ServerContext* context, const SchedulerInfoRequest* request, SchedulerInfoReply* reply) override;
  Status KillObjStore(ServerContext* context, const KillObjStoreRequest* request, AckReply* reply) override;
  Status KillWorker(ServerContext* context, const KillWorkerRequest* request, AckReply* reply) override;

  // ask an object store to send object to another object store. This returns
  // true if the delivery was successfully started. It returns false if the
  // delivery did not successfully start. If the from object store dies during
  // the delivery, it is the responsibility of the to object store to initiate a
  // new delivery with the scheduler.
  bool deliver_object(ObjRef objref, ObjStoreId from, ObjStoreId to);
  // assign a task to a worker
  void schedule();
  // execute a task on a worker and ship required object references. This
  // returns true if the task was successfully submitted and false otherwise.
  // TODO(rkn): This shouldn't take a taskid and a task. I was just passing the
  // taskid before and getting the task, but that required acquiring the
  // computation_graph_lock_, which was causing deadlock. fix this.
  bool submit_task(TaskId taskid, const Call& task, bool is_new_task, WorkerId workerid);
  // checks if the dependencies of the task are met
  bool can_run(const Call& task);
  // register a worker and its object store (if it has not been registered yet)
  std::pair<WorkerId, ObjStoreId> register_worker(const std::string& worker_address, const std::string& objstore_address);
  // register a new object with the scheduler and return its object reference
  ObjRef register_new_object();
  // register the location of the object reference in the object table
  void add_location(ObjRef objref, ObjStoreId objstoreid);
  // indicate that objref is a canonical objref
  void add_canonical_objref(ObjRef objref);
  // get object store associated with a workerid
  ObjStoreId get_store(WorkerId workerid);
  // register a function with the scheduler
  void register_function(const std::string& name, WorkerId workerid, size_t num_return_vals);
  // get information about the scheduler state
  void get_info(const SchedulerInfoRequest& request, SchedulerInfoReply* reply);
private:
  // pick an objectstore that holds a given object (needs protection by objtable_lock_)
  ObjStoreId pick_objstore(ObjRef objref);
  // checks if objref is a canonical objref
  bool is_canonical(ObjRef objref);
  // For canonical objrefs, this is true if the corresponding object is in some
  // object store. For non-canonical objrefs, this is true if the objref is
  // aliased in target_objrefs_.
  bool already_present(ObjRef objref);

  void perform_pulls();
  // schedule tasks using the naive algorithm
  void schedule_tasks_naively();
  // schedule tasks using a scheduling algorithm that takes into account data locality
  void schedule_tasks_location_aware();
  void perform_notify_aliases();

  // checks if aliasing for objref has been completed
  bool has_canonical_objref(ObjRef objref);
  // get the canonical objref for an objref
  ObjRef get_canonical_objref(ObjRef objref);
  // attempt to notify the objstore about potential objref aliasing, returns true if successful, if false then retry later
  bool attempt_notify_alias(ObjStoreId objstoreid, ObjRef alias_objref, ObjRef canonical_objref);
  // tell all of the objstores holding canonical_objref to deallocate it
  void deallocate_object(ObjRef canonical_objref);
  // increment the ref counts for the object references in objrefs
  void increment_ref_count(std::vector<ObjRef> &objrefs);
  // decrement the ref counts for the object references in objrefs
  void decrement_ref_count(std::vector<ObjRef> &objrefs);
  // Find all of the object references which are upstream of objref (including objref itself). That is, you can get from everything in objrefs to objref by repeatedly indexing in target_objrefs_.
  void upstream_objrefs(ObjRef objref, std::vector<ObjRef> &objrefs);
  // Find all of the object references that refer to the same object as objref (as best as we can determine at the moment). The information may be incomplete because not all of the aliases may be known.
  void get_equivalent_objrefs(ObjRef objref, std::vector<ObjRef> &equivalent_objrefs);
  // Recover if an object store dies.
  void recover_from_failed_objstore(ObjStoreId objstoreid);
  // Recover if a worker dies.
  void recover_from_failed_worker(WorkerId workerid);
  // Kill a particular object store.
  void kill_objstore(ObjStoreId objstoreid);
  // Kill a particular worker.
  void kill_worker(WorkerId workerid);
  // Lock all of the scheduler's data structues.
  void acquire_all_locks();
  // Unlock all of the scheduler's data structures.
  void release_all_locks();
  // The computation graph is mostly used for fault tolerance.
  ComputationGraph computation_graph_;
  std::mutex computation_graph_lock_;
  // Vector of all workers registered in the system. Their index in this vector
  // is the workerid.
  std::vector<WorkerHandle> workers_;
  std::mutex workers_lock_;
  // Vector of all workers that are currently idle.
  std::vector<WorkerId> avail_workers_;
  std::mutex avail_workers_lock_;
  // current_tasks_[workerid].first is the taskid of the task that the worker
  // with id workerid is currently executing, if the worker is not executing
  // anything or if the worker is a driver, then this equals NO_TASK.
  // current_tasks_[workerid].second.first is true if this is a brand new task
  // and false if this task is being re-executed because of fault tolerance.
  // current_tasks_[workerid].second.second is the number of remote calls or
  // push calls that the currently executing task on the worker with id workerid
  // has sent to the scheduler. This is relevant only for fault tolerance
  // because the computation graph needs to know which object references to
  // reuse.
  std::vector<std::pair<TaskId, std::pair<bool, size_t> > > current_tasks_;
  // TODO(rkn): current_tasks_ and avail_workers_ contain some of the same
  // information so in order for them to be consistent with one another, they
  // should use the same lock. Actually, current_tasks_ should just be a field
  // of workers_ basically.
  std::mutex current_tasks_lock_;
  // Vector of all object stores registered in the system. Their index in this
  // vector is the objstoreid.
  std::vector<ObjStoreHandle> objstores_;
  grpc::mutex objstores_lock_;
  // Mapping from an aliased objref to the objref it is aliased with. If an
  // objref is a canonical objref (meaning it is not aliased), then
  // target_objrefs_[objref] == objref. For each objref, target_objrefs_[objref]
  // is initialized to UNITIALIZED_ALIAS and the correct value is filled later
  // when it is known.
  std::vector<ObjRef> target_objrefs_;
  std::mutex target_objrefs_lock_;
  // This data structure maps an objref to all of the objrefs that alias it (there could be multiple such objrefs).
  std::vector<std::vector<ObjRef> > reverse_target_objrefs_;
  std::mutex reverse_target_objrefs_lock_;
  // Mapping from canonical objref to list of object stores where the object is stored. Non-canonical (aliased) objrefs should not be used to index objtable_.
  std::vector<std::vector<ObjStoreId> > objtable_;
  std::mutex objtable_lock_;
  // Hash map from function names to workers where the function is registered.
  // TODO(rkn): FnInfo is defined in orchestra.h, but it should probably be defined here.
  std::unordered_map<std::string, FnInfo> fntable_;
  std::mutex fntable_lock_;
  // List of pending tasks. The first component is the TaskId. The second
  // component is true if the task is being sent to a worker for the first time
  // and false if the task has already been sent to a worker before (it doesn't
  // have to have completed).
  std::deque<std::pair<TaskId, bool> > task_queue_;
  std::mutex task_queue_lock_;
  // List of pending pull calls.
  std::vector<std::pair<WorkerId, ObjRef> > pull_queue_;
  std::mutex pull_queue_lock_;
  // List of pending alias notifications. Each element consists of (objstoreid, (alias_objref, canonical_objref)).
  std::vector<std::pair<ObjStoreId, std::pair<ObjRef, ObjRef> > > alias_notification_queue_;
  std::mutex alias_notification_queue_lock_;
  // Reference counts. Currently, reference_counts_[objref] is the number of existing references
  // held to objref. This is done for all objrefs, not just canonical_objrefs. This data structure completely ignores aliasing.
  std::vector<RefCount> reference_counts_;
  std::mutex reference_counts_lock_;
  // contained_objrefs_[objref] is a vector of all of the objrefs contained inside the object referred to by objref
  std::vector<std::vector<ObjRef> > contained_objrefs_;
  std::mutex contained_objrefs_lock_;
  // the scheduling algorithm that will be used
  SchedulingAlgorithmType scheduling_algorithm_;
};

#endif
