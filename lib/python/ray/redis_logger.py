import logging
import redis
import time

import config


# 'entity_type', 'entity_id', 'related_entity_ids', and 'event_type' can be specified in a dictionary
# passed into Python logging calls under the kwarg 'extra'.
# Ex: logger.info("Logging message", extra={'entity_type': 'TASK', 'related_entity_ids': [3, 5, 6]})
REDIS_RECORD_FIELDS = ['log_level',
                       'entity_type',
                       'entity_id',
                       'related_entity_ids',
                       'event_type',
                       'origin',
                       'message']

RAY_FUNCTION = 'FUNCTION'
RAY_OBJECT = 'OBJECT'
RAY_TASK = 'TASK'

class RedisHandler(logging.Handler):
  def __init__(self, origin_type, address, redis_host, redis_port):
    logging.Handler.__init__(self)
    self.origin = "{origin_type}:{address}".format(origin_type=origin_type,
                                                   address=address)
    self.table = 'log'
    self.redis = redis.StrictRedis(host=redis_host,
                                   port=redis_port)

  def check_connected(self):
    try:
      self.redis.ping()
      return True
    except redis.ConnectionError:
      return False

  def emit(self, record):
    # Key is <tablename>:<timestamp>.
    timestamp = "{0:.6f}".format(time.time())
    key = "{table}:{origin}:{timestamp}".format(table=self.table,
                                                origin=self.origin,
                                                timestamp=timestamp)
    record_dict = {}
    for field in REDIS_RECORD_FIELDS:
      record_dict[field] = getattr(record, field, '')
    related_entity_ids = [str(entity_id) for entity_id in getattr(record, 'related_entity_ids', [])]
    record_dict['related_entity_ids'] = ' '.join(related_entity_ids)
    record_dict['log_level'] = logging.getLevelName(self.level)
    record_dict['timestamp'] = timestamp
    self.redis.hmset(key, record_dict)
