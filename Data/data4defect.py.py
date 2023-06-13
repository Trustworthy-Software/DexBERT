import os
import os.path as osp

from sklearn.utils import shuffle
from disassemble import Disassemble
from data4malice import SmaliClassInstructionGenerator
from tqdm import tqdm

def Smalis2Txt(TxtRootDir, SmaliDir):
    '''
    This method extracts Smali Classes in different smali files and save them in a txt file.
    SmaliRootDir: is the disassembled directory from dex files in an APK.
    TxtRootDir: is the root directory to save generated txt files.
    '''
    ApkName = SmaliDir.split('/')[-1] if SmaliDir.split('/')[-1] else SmaliDir.split('/')[-2]
    ApkRootName = SmaliDir.split('/')[-2] if SmaliDir.split('/')[-1] else SmaliDir.split('/')[-3]
    ApkRootDir = osp.join(TxtRootDir, ApkRootName)
    if not osp.exists(ApkRootDir):
        os.makedirs(ApkRootDir)
    with open(osp.join(ApkRootDir, ApkName+'.txt'), 'w') as f:
        for cls in SmaliClassInstructionGenerator(SmaliDir, OnlyFunc=False):
            f.write('ClassName: ' + cls.name + '\n')
            for method in cls.methods:
                f.write('MethodName: ' + method.name + '\n')
                for instruction in method.instructions:
                    f.write(instruction+'\n')
            # for api_name in cls.api_names:
            #     f.write(api_name+'\n')
            f.write('\n')
            f.write('ClassEnd\n\n')


''' generate Smali Txt Files '''

ApkRootDir = 'data/defective_apks'
SmaliTxtDir = 'data/defective_txts'

ApkFileList = []
for root, _, files in os.walk(ApkRootDir, topdown=False):
    ApkFileList = ApkFileList + [osp.join(root, x) for x in files if x.endswith('.apk')]

for apk_path in tqdm(ApkFileList):
    smali_dir = apk_path.split('.apk')[0]
    Disassemble(apk_path, smali_dir)
    Smalis2Txt(SmaliTxtDir, smali_dir)

print('Smali Files Generated!')


''' generate sample lists '''

project_list = ['AnkiDroid', 'FBReader', 'andlytics', 'bankdroid', 'boardgamegeek', 'chess', 'connectbot', 'k9Mail', 'wikipedia', 'yaaic']
ApkRootDir = 'data/defective_apks'
WPDP_ListDir = 'data/WPDP'

for ProjetcName in project_list:
    SmaliFileList = []
    for root, _, files in os.walk(osp.join(ApkRootDir, ProjetcName), topdown=False):
        prefix = root.split(ProjetcName+'/')[-1]
        SmaliFileList = SmaliFileList + [osp.join(ProjetcName, prefix, x) for x in files if x.endswith('.smali')]

    SmaliFileList = shuffle(SmaliFileList)
    list_length = int(len(SmaliFileList)/5)

    list_dir = osp.join(WPDP_ListDir, ProjetcName)
    if not osp.exists(list_dir):
        os.makedirs(list_dir)
    for i in range(5):
        with open(osp.join(list_dir, 'random_part_'+str(i)+'.txt'), 'w') as f:
            if i==4:
                for name in SmaliFileList[list_length*i:]:
                    f.write(name+'\n')
            else:
                for name in SmaliFileList[list_length*i:list_length*(i+1)]:
                    f.write(name+'\n')
        
print('WPDP Lists Generated!')

''' split label files '''

root_dir = 'data/defect_labels'

txt_list = [x for x in os.listdir(root_dir) if x.endswith('.txt')]

for txt_name in txt_list:
    project_dir = osp.join(root_dir, txt_name.split('.txt')[0])
    if not osp.exists(project_dir):
        os.makedirs(project_dir)
    smali_names = sorted([x.strip() for x in open(osp.join(root_dir, txt_name), 'r').readlines()])
    app_version = smali_names[0].split('/')[1]
    version_file = open(osp.join(project_dir, app_version+'.txt'), 'w')
    for smali_name in smali_names:
        if not smali_name.split('/')[0] == txt_name.split('.txt')[0]:
            continue
        if smali_name.split('/')[1] == app_version:
            version_file.write('/'.join(smali_name.split('/')[2:]).split('.smali')[0]+'\n')
        else:
            version_file.close()
            app_version = smali_name.split('/')[1]
            version_file = open(osp.join(project_dir, app_version+'.txt'), 'w')
            version_file.write('/'.join(smali_name.split('/')[2:]).split('.smali')[0]+'\n')
    version_file.close()

''' generate data file'''

smali_txt_dir = 'data/defective_txts'
label_file_dir = 'data/defect_labels'
data_file_save_dir = 'data/defect_data_lists'  # generated data lists

project_list = ['AnkiDroid', 'FBReader', 'andlytics', 'bankdroid', 'boardgamegeek', 'chess', 'connectbot', 'k9Mail', 'wikipedia', 'yaaic']

for project in project_list:
    data_file = open(osp.join(data_file_save_dir, project+'.txt'), 'w')
    version_files = [x for x in os.listdir(osp.join(smali_txt_dir, project)) if x.endswith('.txt')]
    for version_name in version_files:
        data_line = osp.join(smali_txt_dir, project, version_name)
        label_line = osp.join(label_file_dir, project, version_name)
        data_file.write(data_line+','+label_line+'\n')
    data_file.close()