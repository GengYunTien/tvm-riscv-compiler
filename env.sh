# Usage: source env.sh
export TVM_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/3rdparty/tvm"
export TVM_LIBRARY_PATH="$TVM_HOME/build"
export PYTHONPATH="$TVM_HOME/python"
