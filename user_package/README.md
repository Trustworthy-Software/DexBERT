This tool can help you generate smali class embedding vectors of DexBERT for a given APK.

Before running the code, you need to:

    - specify your personal AndroZoo API Key (which can be obtained from https://androzoo.uni.lu/) in utils.py

    - set up your python env according to requirements.txt (minicoda is recommended).

    - download [pretrained_dexbert_model_steps_604364.pt](https://drive.google.com/file/d/1z6aZQXT1dS6wX1JgPnWJVS_e6Td2sBPg/view?usp=sharing) to current directory.

    - set the path of apk hash list and the directory where you want to save the embedding vectors in GenDexBertEmbeddings.py (parameters 'root_dir', and 'src_data_list')

    - check which GPU has at least 6G free memory and set its id with ```os.environ["CUDA_VISIBLE_DEVICES"]= 'gpu_id'``` in GenDexBertEmbeddings.py

run ```python GenDexBertEmbeddings.py```