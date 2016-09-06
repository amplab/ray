package(default_visibility = ["//visibility:public"])

cc_library(
    name = "flatbuffers",
    srcs = [
        "src/idl_gen_text.cpp",
        "src/idl_parser.cpp",
        "src/reflection.cpp",
        "src/util.cpp",
    ],
    hdrs = glob(["include/flatbuffers/*.h"]),
    includes = ["include"],
)

cc_binary(
    name = "flatc",
    srcs = [
        "src/flatc.cpp",
        "src/idl_gen_cpp.cpp",
        "src/idl_gen_fbs.cpp",
        "src/idl_gen_general.cpp",
        "src/idl_gen_go.cpp",
        "src/idl_gen_js.cpp",
        "src/idl_gen_php.cpp",
        "src/idl_gen_python.cpp",
    ],
    deps = [":flatbuffers"],
)
