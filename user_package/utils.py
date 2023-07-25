import os
import os.path as osp
import sys
import torch
import logging
import random
import numpy as np

API_key = ""  # please obtain your personal key on https://androzoo.uni.lu/

def DownloadApk(ApkFile):
    '''
    To download ApkFile that doesn't exist.

    :param String ApkFile: absolute path of the ApkFile
    '''

    if osp.exists(ApkFile):
        pass
    else:
        SaveDir, ApkName = osp.dirname(ApkFile), osp.basename(ApkFile)
        Hash = ApkName.split('.')[0]
        os.system("cd {} && curl -O --remote-header-name -G -d apikey={} -d sha256={} https://androzoo.uni.lu/api/download > /dev/null".format(
            SaveDir, API_key, Hash))

def Disassemble(ApkPath, OutDir):
    '''
    To disassemble Dex bytecode in a given Apk file into smali code.
    Java version: "11.0.11" 2021-04-20
    The baksmali tool baksmali-2.5.2.jar was downloaded on: https://bitbucket.org/JesusFreke/smali/downloads/
    '''
    os.system("java -jar {} disassemble {} -o {}".format(osp.join(sys.path[0], 'baksmali-2.5.2.jar'), ApkPath, OutDir)) 

def get_device():
    "get device (CPU or GPU)"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("%s (%d GPUs)" % (device, n_gpu))
    return device

def truncate_tokens(tokens, max_len):
    while True:
        if len(tokens) <= max_len:
            break
        else:
            tokens.pop()

def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)

def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

def find_sublist(haystack, needle):
    """Return the index at which the sequence needle appears in the
    sequence haystack, or -1 if it is not found, using the Boyer-
    Moore-Horspool algorithm. The elements of needle and haystack must
    be hashable.
    https://codereview.stackexchange.com/questions/19627/finding-sub-list
    """
    h = len(haystack)
    n = len(needle)
    skip = {needle[i]: n - i - 1 for i in range(n - 1)}
    i = n - 1
    while i < h:
        for j in range(n):
            if haystack[i - j] != needle[-j - 1]:
                i += skip.get(haystack[i], n)
                break
        else:
            return i - n + 1
    return -1

def truncate_tokens_pair(tokens_a, tokens_b, max_len):
    while True:
        if len(tokens_a) + len(tokens_b) <= max_len:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def truncate_tokens(tokens, max_len):
    while True:
        if len(tokens) <= max_len:
            break
        else:
            tokens.pop()

def get_random_word(vocab_words):
    i = random.randint(0, len(vocab_words)-1)
    return vocab_words[i]

def get_random_word_list(vocab_words: list, random_length: int):
    random_list = []
    for _ in range(random_length):
        i = random.randint(0, len(vocab_words)-1)
        random_list.append(vocab_words[i])
    return random_list

def get_logger(name, log_path):
    "get logger"
    logger = logging.getLogger(name)
    fomatter = logging.Formatter(
        '[ %(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')

    if not os.path.isfile(log_path):
        f = open(log_path, "w+")

    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(fomatter)
    logger.addHandler(fileHandler)

    #streamHandler = logging.StreamHandler()
    #streamHandler.setFormatter(fomatter)
    #logger.addHandler(streamHandler)

    logger.setLevel(logging.DEBUG)
    return logger
