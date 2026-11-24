import subprocess
import os 
import yaml
import natsort
from llamafactory.train.tuner import run_exp
# /home/user_home/qiankunjie/LLaMA-Factory/src/llamafactory

#读取指定目录里边的文件夹 
def list_directories(folder_path):
    dir_list = natsort.natsorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
    return dir_list[0:] #

print('===============sft_addlabel===============')
root_dir = "path/to"
# check_point保存路径
check_points_dir = "./result_save/C-MTCSD/Pre-made Meals/qwen3_noInstance_LK-local-multi-layers-SPKD-dual-topKL_instruct_prompt_1_CFF1024_0402_seed42/"
# 数据路径
dataset_dir = root_dir + "/datasets/C-MTCSD/sft_addlabel_noInstance_CUSTOM-LOSS_instruct_0522/Pre-made Meals"
# 预测保存路径
output_dir = "./result_save/predict/Qwen3-8B_C-MTCSD_Pre-made_Meals_noInstance_LK-local-multi-layers-SPKD-dual-topKL_instruct_prompt_1_CFF1024_0402_seed42_1112"

#基本的配置模板。
base_config_path = "C-MTCSD_Pre-made_Meals_predict.yaml"
base_config  = yaml.load(open(base_config_path, 'rb'), Loader=yaml.Loader)

# 遍历check_point 保存的目录里边的 checkpoint
for each_check_points in list_directories(check_points_dir):
    # 以checkpoint的文件夹
    if each_check_points.startswith('checkpoint'):
        # print('each_check_points: ', each_check_points)
        #这里是覆盖基本配置里边的变量，如果不复制就读取配置里边的变量
        base_config['eval_dataset'] = "Pre-made Meals_test"
        base_config['per_device_eval_batch_size'] = 1
        base_config['do_sample'] = False
        base_config['temperature'] = 1.0
        base_config['top_p'] = 1.0
        base_config['cutoff_len'] = 1024
        base_config['dataset_dir'] = dataset_dir
        base_config['adapter_name_or_path'] = check_points_dir + "/" + each_check_points
        base_config['output_dir'] = output_dir +"/" + each_check_points + "-greedy-decoding-defaults-bs1"
        # 穿进去开始推理
        run_exp(base_config)
