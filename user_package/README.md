This tool can help you generate smali class embedding vectors of DexBERT for a given APK.

Before running the code, you need to:

    - specify your personal AndroZoo API Key (which can be obtained from https://androzoo.uni.lu/) in utils.py

    - set up your python env according to requirements.txt (minicoda is recommended).

    - set the path of apk hash list and the directory where you want to save the embedding vectors in GenDexBertEmbeddings.py (parameters 'root_dir', and 'src_data_list')

    - check which GPU has at least 6G free memory and set its id with ```os.environ["CUDA_VISIBLE_DEVICES"]= 'gpu_id'``` in GenDexBertEmbeddings.py

run ```python GenDexBertEmbeddings.py```