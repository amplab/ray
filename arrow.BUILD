package(default_visibility = ["//visibility:public"])

prefix_dir = "cpp/src"

genrule(
    name = "arrow_ipc_message",
    srcs = ["format/Message.fbs"],
    outs = [prefix_dir + "/arrow/ipc/Message_generated.h"],
    cmd = "$(location @flatbuffers//:flatc) -c -o $(@D) $(SRCS)",
    tools = ["@flatbuffers//:flatc"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "arrow_ipc_message_generated",
    hdrs = [prefix_dir + "/arrow/ipc/Message_generated.h"],
)

cc_library(
    name = "arrow",
    srcs = [
        prefix_dir + "/arrow/array.cc",
        prefix_dir + "/arrow/builder.cc",
        prefix_dir + "/arrow/column.cc",
        prefix_dir + "/arrow/schema.cc",
        prefix_dir + "/arrow/table.cc",
        prefix_dir + "/arrow/type.cc",
        prefix_dir + "/arrow/ipc/adapter.cc",
        prefix_dir + "/arrow/ipc/memory.cc",
        prefix_dir + "/arrow/ipc/metadata.cc",
        prefix_dir + "/arrow/ipc/metadata-internal.cc",
        prefix_dir + "/arrow/types/construct.cc",
        prefix_dir + "/arrow/types/decimal.cc",
        prefix_dir + "/arrow/types/json.cc",
        prefix_dir + "/arrow/types/list.cc",
        prefix_dir + "/arrow/types/primitive.cc",
        prefix_dir + "/arrow/types/string.cc",
        prefix_dir + "/arrow/types/struct.cc",
        prefix_dir + "/arrow/types/union.cc",
        prefix_dir + "/arrow/util/bit-util.cc",
        prefix_dir + "/arrow/util/buffer.cc",
        prefix_dir + "/arrow/util/memory-pool.cc",
        prefix_dir + "/arrow/util/status.cc",
    ],
    hdrs = glob([prefix_dir + "/arrow/**/*.h"]),
    includes = [prefix_dir],
    deps = [
        "//:arrow_ipc_message_generated",
        "@boost_archive//:boost",
        "@boost_archive//:filesystem",
        "@flatbuffers//:flatbuffers",
    ],
)
