import os
import os.path as osp
import requests
import random
from collections import defaultdict
from androguard.core.bytecodes import apk
from tqdm import tqdm

from data4malice import SmaliClassInstructionGenerator
from disassemble import Disassemble


def download_apk(api_key, apk_hash, output_folder):
    url = f'https://androzoo.uni.lu/api/download?apikey={api_key}&sha256={apk_hash}'
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        apk_path = os.path.join(output_folder, f'{apk_hash}.apk')
        with open(apk_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f'Successfully downloaded: {apk_hash}')
    else:
        print(f'Failed to download: {apk_hash}')

def download_apks_from_txt(api_key, txt_file, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    with open(txt_file, 'r') as f:
        for line in f:
            apk_hash = line.strip()
            download_apk(api_key, apk_hash, output_folder)

def extract_android_manifest(apk_file):
    a = apk.APK(apk_file)
    manifest_data = a.get_android_manifest_xml()
    return manifest_data


def parse_android_manifest(manifest_data):
    result = {}

    for activity in manifest_data.iter('activity'):
        class_name = activity.attrib['{http://schemas.android.com/apk/res/android}name']
        result[class_name] = 'Activity'

    for service in manifest_data.iter('service'):
        class_name = service.attrib['{http://schemas.android.com/apk/res/android}name']
        result[class_name] = 'Service'

    for receiver in manifest_data.iter('receiver'):
        class_name = receiver.attrib['{http://schemas.android.com/apk/res/android}name']
        result[class_name] = 'BroadcastReceiver'

    for provider in manifest_data.iter('provider'):
        class_name = provider.attrib['{http://schemas.android.com/apk/res/android}name']
        result[class_name] = 'ContentProvider'

    return result


def save_to_txt(output_file, component_data):
    with open(output_file, 'w') as f:
        for class_name, component_type in component_data.items():
            f.write(f'{class_name}:{component_type}\n')


def process_apk(apk_file, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    manifest_data = extract_android_manifest(apk_file)
    component_data = parse_android_manifest(manifest_data)
    apk_name = os.path.splitext(os.path.basename(apk_file))[0]
    output_file = os.path.join(output_folder, f'{apk_name}_output.txt')
    save_to_txt(output_file, component_data)
    return component_data

def gen_component_type(apk_folder, output_folder):
    component_count = {'Activity': 0, 'Service': 0, 'BroadcastReceiver': 0, 'ContentProvider': 0}

    for file in tqdm(os.listdir(apk_folder)):
        if file.endswith(".apk"):
            try:
                apk_file = os.path.join(apk_folder, file)
                component_data = process_apk(apk_file, output_folder)
                for component_type in component_data.values():
                    component_count[component_type] += 1
            except:
                continue

    print("Component Count:")
    for component_type, count in component_count.items():
        print(f'{component_type}: {count}')

def filter_activities(input_file, output_file, percentage=0.25):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    activity_lines = [line for line in lines if 'Activity' in line]
    non_activity_lines = [line for line in lines if 'Activity' not in line]

    selected_activity_lines = random.sample(activity_lines, int(len(activity_lines) * percentage))

    with open(output_file, 'w') as f:
        for line in selected_activity_lines + non_activity_lines:
            f.write(line)

    return len(selected_activity_lines), non_activity_lines


def process_output_files(input_folder, output_folder, percentage=0.25):
    os.makedirs(output_folder, exist_ok=True)

    final_count = defaultdict(int)

    for file in os.listdir(input_folder):
        if file.endswith('_output.txt'):
            input_file = os.path.join(input_folder, file)
            output_file = os.path.join(output_folder, file)
            activity_count, non_activity_lines = filter_activities(input_file, output_file, percentage)

            final_count['Activity'] += activity_count

            for line in non_activity_lines:
                _, component_type = line.strip().split(':')
                final_count[component_type] += 1

    print("Final Component Count:")
    for component_type, count in final_count.items():
        print(f'{component_type}: {count}')

def Smalis2Txt(TxtRootDir, SmaliDir, Component_type_dir, only_keep_func_name=False):
    '''
    This method extracts Smali Classes in different smali files and save them in a txt file.
    SmaliRootDir: is the disassembled directory from dex files in an APK.
    TxtRootDir: is the root directory to save generated txt files.
    '''
    ApkName = SmaliDir.split('/')[-1] if SmaliDir.split('/')[-1] else SmaliDir.split('/')[-2]
    type_file = open(osp.join(Component_type_dir, ApkName+'_output.txt'), 'r').readlines()
    class_dic = {}
    for line in type_file:
        class_dic[line.split(':')[0]] = line.split(':')[1].strip()
    with open(osp.join(TxtRootDir, ApkName+'.txt'), 'w') as f:
        for cls in SmaliClassInstructionGenerator(SmaliDir, OnlyFunc=only_keep_func_name):
            if not cls.name.replace('/', '.') in class_dic:
                continue
            # import ipdb; ipdb.set_trace()
            f.write('ClassName: ' + cls.name + '\n')
            if only_keep_func_name:
                for api_name in cls.api_names:
                    f.write(api_name+'\n')
            else:
                for method in cls.methods:
                    f.write('MethodName: ' + method.name + '\n')
                    for instruction in method.instructions:
                        f.write(instruction+'\n')
                    f.write('\n')
            f.write('\n')
            f.write('ClassEnd\n\n')

if __name__ == "__main__":

    api_key = ""  # One could apply for it on https://androzoo.uni.lu/
    
    test_hash_list = 'data/component_classification/test/hash_list.txt'
    test_apk_folder = 'data/component_classification/test/apks'
    test_component_types_folder = 'data/component_classification/test/component_types'
    train_hash_list = 'data/component_classification/train/hash_list.txt'
    train_apk_folder = 'data/component_classification/train/apks'
    train_component_types_folder = 'data/component_classification/train/component_types'
    train_filtered_component_types_folder = 'data/component_classification/train/component_types_filtered'

    test_data_file = 'data/component_classification/test/data_file.txt'
    train_data_file = 'data/component_classification/train/data_file.txt'
    
    download_apks_from_txt(api_key=api_key,
                           txt_file=test_hash_list, output_folder=test_apk_folder)
    download_apks_from_txt(api_key=api_key,
                           txt_file=train_hash_list, output_folder=train_apk_folder)
    
    gen_component_type(test_apk_folder, test_component_types_folder)
    gen_component_type(train_apk_folder, train_component_types_folder)
    
    # filtering is only for training set to avoid severe data imbalance
    process_output_files(train_component_types_folder, train_filtered_component_types_folder, percentage=0.25)

    for sample in tqdm(open(test_hash_list, 'r').readlines()):
        hash = sample.strip()
        apk_path = osp.join(test_apk_folder, hash.strip().upper()+'.apk')
        smali_dir = osp.join(test_apk_folder, hash)
        Disassemble(apk_path, smali_dir)
        Smalis2Txt(test_apk_folder, smali_dir, test_component_types_folder, only_keep_func_name=False)

    for sample in tqdm(open(train_hash_list, 'r').readlines()):
        hash = sample.strip()
        apk_path = osp.join(train_apk_folder, hash.strip().upper()+'.apk')
        smali_dir = osp.join(train_apk_folder, hash)
        Disassemble(apk_path, smali_dir)
        Smalis2Txt(train_apk_folder, smali_dir, train_filtered_component_types_folder, only_keep_func_name=False)
    
    data_file = open(test_data_file, 'w')
    hash_list = open(test_hash_list, 'r').readlines()
    for hash in hash_list:
        hash = hash.strip()
        data_file.write(os.path.join(test_apk_folder, hash+'.txt')+','+os.path.join(test_component_types_folder, hash+'_output.txt'+'\n'))
    data_file.close()

    data_file = open(train_data_file, 'w')
    hash_list = open(train_hash_list, 'r').readlines()
    for hash in hash_list:
        hash = hash.strip()
        data_file.write(os.path.join(train_apk_folder, hash+'.txt')+','+os.path.join(train_filtered_component_types_folder, hash+'_output.txt'+'\n'))
    data_file.close()
