package(default_visibility = ["//visibility:public"])

cc_library(
    name = "numbuf_cc",
    srcs = [
        "cpp/src/numbuf/dict.cc",
        "cpp/src/numbuf/sequence.cc",
        "cpp/src/numbuf/tensor.cc",
    ],
    hdrs = [
        "cpp/src/numbuf/dict.h",
        "cpp/src/numbuf/sequence.h",
        "cpp/src/numbuf/tensor.h",
    ],
    includes = ["cpp/src"],
    deps = [
        "@arrow//:arrow",
    ],
)

cc_binary(
    name = "libnumbuf.so",
    srcs = [
        "python/src/pynumbuf/adapters/numpy.cc",
        "python/src/pynumbuf/adapters/numpy.h",
        "python/src/pynumbuf/adapters/python.cc",
        "python/src/pynumbuf/adapters/python.h",
        "python/src/pynumbuf/adapters/scalars.h",
        "python/src/pynumbuf/memory.h",
        "python/src/pynumbuf/numbuf.cc",
    ],
    linkshared = 1,
    linkstatic = 1,
    deps = [
        ":numbuf_cc",
        "@arrow//:arrow",
        "@python//:numpy_headers",
        "@python//:python_headers",
    ],
)
