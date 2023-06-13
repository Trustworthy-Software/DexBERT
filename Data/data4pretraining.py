'''
download
disassemble
remove original apk
'''
import os
import os.path as osp
import argparse
import logging

import time
import pickle
import time
import psutil
import json
import multiprocessing as mp
from datetime import timedelta
from generate_vocab import gen_vocab

from disassemble import Disassemble
from instruction_generator import SmaliInstructionGenerator, ClassDictionary

API_key = "************************"  # One could apply for it on https://androzoo.uni.lu/

'''logger'''
logging.basicConfig(level=logging.INFO,filename="LogFile.log",filemode="a",
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')
ErrorHandler = logging.StreamHandler()
ErrorHandler.setLevel(logging.ERROR)
ErrorHandler.setFormatter(logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'))
logger = logging.getLogger()
logger.addHandler(ErrorHandler)

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

def ProcessingDataForGetApkData(ApkFile):
    try:
        StartTime = time.time()
        DownloadApk(ApkFile)
        logger.info("Start to process " + ApkFile + "...")
        print("Start to process " + ApkFile + "...")
        
        # generate instructions from apk
        SmaliDir = osp.splitext(ApkFile)[0]
        Disassemble(ApkFile, SmaliDir)
        InstructionFile = SmaliDir+'.txt'
        with open(InstructionFile, 'w') as f:
            f.write(InstructionFile.split('/')[-1]+'\n')
            for cls in SmaliInstructionGenerator(SmaliRootDir=SmaliDir, flag='class'):
                f.write('ClassName: ' + cls.name + '\n')
                for method in cls.methods:
                    f.write('MethodName: ' + method.name + '\n')
                    for instruction in method.instructions:
                        f.write(instruction+'\n')
                    f.write('\n')
        if osp.exists(ApkFile):
            os.system('rm {}'.format(ApkFile))
        if osp.exists(SmaliDir):
            os.system('rm -r {}'.format(SmaliDir))
    except Exception as e:
        FinalTime = time.time()
        logger.error(e)
        logger.error(ApkFile + " processing failed in " + str(FinalTime - StartTime) + "s...")
        print(ApkFile + " processing failed in " + str(FinalTime - StartTime) + "s...")
        if osp.exists(ApkFile):
            os.system('rm {}'.format(ApkFile))
        return ApkFile, False
    else:
        FinalTime = time.time()
        logger.info(ApkFile + " processed successfully in " + str(FinalTime - StartTime) + "s")
        print(ApkFile + " processed successfully in " + str(FinalTime - StartTime) + "s")
        return ApkFile, True


def GetApkData(ProcessNumber, Sha256ListPath, ApkDirectoryPath):
    '''
    Get Apk data dictionary for all Apk files under ApkDirectoryPath and store them in ApkDirectoryPath
    Used for next step's classification

    :param string Sha256ListPath: absolute path of the pickle/txt file that contains Apk sha256s.
    :param string ApkDirectoryPath: absolute path of the directory to save Apk features.
    '''
    HashList = []
    if Sha256ListPath.endswith('.pkl'):
        with open(Sha256ListPath, 'rb') as f:
            HashList = pickle.load(f)
    elif Sha256ListPath.endswith('.txt'):
        with open(Sha256ListPath, 'r') as f:
            HashList = f.readlines()
    else:
        print('Wrong Sha256ListPath !!!')
        exit()
    ApkFileList = [osp.join(ApkDirectoryPath, hash.strip().upper()+'.apk') for hash in HashList]
    if not osp.exists(ApkDirectoryPath):
        os.makedirs(ApkDirectoryPath)

    time_start = time.time()
    pool = mp.Pool(int(ProcessNumber))
    for ApkFile in ApkFileList:
        if osp.exists(osp.splitext(ApkFile)[0]) and osp.exists(osp.splitext(ApkFile)[0] + ".txt"):
            pass
        else:
            pool.apply_async(ProcessingDataForGetApkData, args=(ApkFile,))
    pool.close()
    pool.join()
    with open(osp.join(ApkDirectoryPath, 'ClassDictionary.json'), 'w') as dicfile:
        json.dump(dict(sorted(ClassDictionary.copy().items(), key=lambda item: item[1], reverse=True)), dicfile, indent=4)
    print("Time cost: ", time.time()-time_start)


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", help="The path to the APK files directory", type=str, required=True) 
    parser.add_argument("-l", "--list", help="The path to the pkl/txt file that contains sha256 list", type=str, required=True)
    parser.add_argument("-cp", "--cpu", help="The number of CPUs to use, default value is the number of logical CPUs in the system", type=int, required=False, default=psutil.cpu_count())
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    Args = parseargs()
    ProcessNumber = Args.cpu
    ApkDirectoryPath = Args.dir
    Sha256ListPath = Args.list
    time_start = time.time()
    GetApkData(ProcessNumber, Sha256ListPath, ApkDirectoryPath)
    os.system("cd {}; find . -name '{}' ! -name {} -exec cat {} + > {}".format(ApkDirectoryPath, '*txt', 'data_file.txt', '{}', 'data_file.txt'))
    os.system("cd {}; find . -name '{}' ! -name {} | xargs rm -f".format(ApkDirectoryPath, '*txt', 'data_file.txt'))
    logger.info("Data File Generated in {} ...".format(str(timedelta(seconds=time.time()-time_start))))

    # generate vocabulary
    time_start = time.time()
    tmp_dir = osp.join(ApkDirectoryPath, 'tmp')
    if not osp.exists(tmp_dir):
        os.makedirs(tmp_dir)
    os.system("rm -rf {}/*".format(tmp_dir))
    os.system("cp {} {}".format(osp.join(ApkDirectoryPath, 'data_file.txt'), tmp_dir))
    gen_vocab(dataset_dir=ApkDirectoryPath, vocab_file=osp.join(ApkDirectoryPath, 'vocab.txt'), vocab_size=10000)
    os.system("rm -rf {}".format(tmp_dir))
    logger.info("vocabulary Generated in {} ...".format(str(timedelta(seconds=time.time()-time_start))))
