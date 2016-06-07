#ifndef QUARTZ_INCLUDE_QUARTZ_H
#define QUARTZ_INCLUDE_QUARTZ_H

#include <vector>
#include <unordered_map>

typedef size_t ObjRef;
typedef size_t WorkerId;
typedef size_t ObjStoreId;
typedef size_t OperationId;

class FnInfo {
  size_t num_return_vals_;
  std::vector<WorkerId> workers_; // `workers_` is a sorted vector
public:
  void set_num_return_vals(size_t num) {
    num_return_vals_ = num;
  }
  size_t num_return_vals() const {
    return num_return_vals_;
  }
  void add_worker(WorkerId workerid) {
    // insert `workerid` into `workers_` so that `workers_` stays sorted
    workers_.insert(std::lower_bound(workers_.begin(), workers_.end(), workerid), workerid);
  }
  size_t num_workers() const {
    return workers_.size();
  }
  const std::vector<WorkerId>& workers() const {
    return workers_;
  }
};

typedef std::vector<std::vector<ObjStoreId> > ObjTable;
typedef std::unordered_map<std::string, FnInfo> FnTable;

#define QUARTZ_VERBOSE -1
#define QUARTZ_INFO 0
#define QUARTZ_DEBUG 1
#define QUARTZ_FATAL 2
#define QUARTZ_REFCOUNT QUARTZ_VERBOSE
#define QUARTZ_ALIAS QUARTZ_VERBOSE

#define QUARTZ_LOG(LEVEL, MESSAGE) \
  if (LEVEL == QUARTZ_VERBOSE) { \
    \
  } else if (LEVEL == QUARTZ_FATAL) { \
    std::cerr << "fatal error occured: " << MESSAGE << std::endl; \
    std::exit(1); \
  } else if (LEVEL == QUARTZ_DEBUG) { \
    \
  } else { \
    std::cout << MESSAGE << std::endl; \
  }

class objstore_not_registered_error : public std::runtime_error
{
public:
  objstore_not_registered_error(const std::string& msg) : std::runtime_error(msg) {}
};

struct slice {
  uint8_t* data;
  size_t len;
};

#endif
