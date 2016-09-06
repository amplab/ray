def _python_configure(repository_ctx):
  # Find Python headers
  result = repository_ctx.execute(
    ["python", "-c",
    "from __future__ import print_function; " +
    "from distutils import sysconfig; " +
    "print(sysconfig.get_python_inc());"])
  if result.stderr:
    fail("\nCouldn't find Python headers:\n%s" % result.stderr)
  repository_ctx.symlink(result.stdout.strip(), "python_headers")

  # Find numpy headers
  result = repository_ctx.execute(
    ["python", "-c",
    "from __future__ import print_function; " +
    "import numpy; " +
    "print(numpy.get_include());"])
  if result.stderr:
    fail("\nCouldn't find numpy headers:\n%s" % result.stderr)
  repository_ctx.symlink(result.stdout.strip(), "numpy_headers")

  repository_ctx.file(
    "BUILD",
    """
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "python_headers",
    hdrs = glob([
        "python_headers/**/*.h"  
    ]),
    includes = ["python_headers"],
)

cc_library(
    name = "numpy_headers",
    hdrs = glob([
        "numpy_headers/**/*.h"  
    ]),
    includes = ["numpy_headers"],
)
""")

python_configure = repository_rule(
  implementation = _python_configure,
  local = True,
)
