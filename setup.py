"""Download the model and datasets into the Hugging Face cache."""
from datasets import load_dataset
from huggingface_hub import snapshot_download

snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct")
load_dataset("derek-thomas/ScienceQA")
load_dataset("HuggingFaceM4/A-OKVQA")
