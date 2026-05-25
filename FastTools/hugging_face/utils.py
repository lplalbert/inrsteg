import transformers
from FastTools.setting.Key import huggingface_cache_path
import os
def download_model(
    model_name, 
    token = os.environ.get("HF_TOKEN", ""),
    cache_dir=None
    ):
    if cache_dir is None:
        cache_dir = huggingface_cache_path
        
    os.system("HF_ENDPOINT=https://hf-mirror.com python ./FastTools/hugging_face/hf_download.py --model {}  --save_dir {} --token {}".format(
        model_name,
        cache_dir,
        token
    ))
    pass

def download_dataset(
    dataset_name,
    token=os.environ.get("HF_TOKEN", ""),
    cache_dir=None
    ):
    if cache_dir is None:
        cache_dir = huggingface_cache_path
    
    os.system("HF_ENDPOINT=https://hf-mirror.com python ./FastTools/hugging_face/hf_download.py --dataset {}  --save_dir {} --token {}".format(
        dataset_name,
        cache_dir,
        token
    ))
    pass
if __name__ == "__main__":
    print(huggingface_cache_path)
    # download_model(
    #     "facebook/opt-125m"
    # )
    
    download_dataset(
        "HuggingFaceH4/ultrachat_200k",
        cache_dir="/home/light_sun/workspace/text/data/.cache"
    )
    pass






    