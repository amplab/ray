load("//:python_configure.bzl", "python_configure")

python_configure(name = "python")

new_http_archive(
    name = "boost_archive",
    build_file = "boost.BUILD",
    sha256 = "a77c7cc660ec02704c6884fbb20c552d52d60a18f26573c9cee0788bf00ed7e6",
    url = "http://pilotfiber.dl.sourceforge.net/project/boost/boost/1.61.0/boost_1_61_0.tar.gz",
)

new_git_repository(
    name = "arrow",
    build_file = "arrow.BUILD",
    commit = "c781ef8564714bb5149e8b5a311673f9162232ac",
    remote = "https://github.com/pcmoritz/arrow.git",
)

new_git_repository(
    name = "numbuf",
    build_file = "numbuf.BUILD",
    commit = "74406e92e4e10c228f2850e72ea835980c779e3b",
    remote = "https://github.com/pcmoritz/numbuf.git",
)

new_git_repository(
    name = "flatbuffers",
    build_file = "flatbuffers.BUILD",
    commit = "5a401aef57460a8a6c5b4e77dacea61a0085bbb5",
    remote = "https://github.com/google/flatbuffers.git",
)

new_git_repository(
    name = "grpc",
    build_file = "grpc.BUILD",
    commit = "2a69139aa7f609e439c24a46754252a5f9d37500",
    init_submodules = 1,
    remote = "https://github.com/grpc/grpc.git",
)

git_repository(
    name = "protobuf",
    commit = "e8ae137c96444ea313485ed1118c5e43b2099cf1",
    remote = "https://github.com/google/protobuf.git",
)

new_git_repository(
    name = "zlib_git",
    build_file = "zlib.BUILD",
    commit = "50893291621658f355bc5b4d450a8d06a563053d",
    remote = "https://github.com/madler/zlib",
)

new_http_archive(
    name = "six_archive",
    build_file = "@protobuf//:six.BUILD",
    sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
    url = "https://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz#md5=34eed507548117b2ab523ab14b2f8b55",
)

bind(
    name = "six",
    actual = "@six_archive//:six",
)

# Needed by gRPC
bind(
    name = "protobuf_clib",
    actual = "@protobuf//:protoc_lib",
)

bind(
    name = "protobuf_compiler",
    actual = "@protobuf//:protoc_lib",
)

# Needed by the protobuf gRPC plugin
bind(
    name = "grpc_cpp_plugin",
    actual = "@grpc//:grpc_cpp_plugin",
)

bind(
    name = "grpc_lib",
    actual = "@grpc//:grpc++_unsecure",
)

bind(
    name = "zlib",
    actual = "@zlib_git//:zlib",
)
