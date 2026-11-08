# Creates the vlmcot conda env and downloads the model and datasets (~18 GB).
#   bash setup.sh                  # downloads go to home directory
#   bash setup.sh /path/hf_cache   # downloads go there instead
set -e
cd "$(dirname "$0")"

# install env
conda env create -y -f env/environment.yml

# optional HF cache location
if [ -n "$1" ]; then
  mkdir -p "$1"
  export HF_HOME="$(realpath "$1")"
  conda env config vars set -n vlmcot HF_HOME="$HF_HOME"
fi

# setup HF/Qwen
conda run -n vlmcot --live-stream python setup.py
