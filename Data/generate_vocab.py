import time
import argparse

import tensorflow as tf
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

'''
This script is used to generate a feature vocabulary file given text files.
The dataset_dir should contain several sub_dirs which contain text files belong to different categories.
The vocab_file is the final generated feature vocabulary file.
'''

pre_defined = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']

def gen_vocab(dataset_dir, vocab_file, vocab_size):

    batch_size = 48
    AUTOTUNE = tf.data.AUTOTUNE

    test_ds = tf.keras.preprocessing.text_dataset_from_directory(
        dataset_dir,
        batch_size=batch_size,
        validation_split=0.05,
        subset='validation',
        seed=42)
    test_ft = test_ds.map(lambda ft, lb: ft)

    bert_tokenizer_params=dict(lower_case=True)
    reserved_tokens=pre_defined

    bert_vocab_args = dict(
        # The target vocabulary size
        vocab_size = vocab_size,
        # Reserved tokens that must be included in the vocabulary
        reserved_tokens=reserved_tokens,
        # Arguments for `text.BertTokenizer`
        bert_tokenizer_params=bert_tokenizer_params,
        # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
        learn_params={},
    )

    StartTime = time.time()
    pt_vocab = bert_vocab.bert_vocab_from_dataset(
        test_ft.batch(batch_size).prefetch(buffer_size=AUTOTUNE),
        **bert_vocab_args
    )
    EndTime = time.time()
    print("bert_vocab_from_dataset costs {} seconds.".format(EndTime - StartTime))

    write_vocab_file(vocab_file, pt_vocab)

def write_vocab_file(filepath, vocab):
      with open(filepath, 'w') as f:
        for token in vocab:
            print(token, file=f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", help="The dir should contain several sub_dirs which contain text files belong to different categories", type=str, required=True) 
    parser.add_argument("-f", "--file", help="The file is the final generated feature vocabulary file.", type=str, required=True)
    parser.add_argument("-s", "--size", help="The target vocabulary size.", type=int, default=10000)
    args = parser.parse_args()

    gen_vocab(dataset_dir=args.dir, vocab_file=args.file, vocab_size=args.size)
