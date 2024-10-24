import os
import argparse
import json

"""
文件名: configs.py

功能:
    1. 用args提供config, 对外有一个函数为get_configs() 输出一个args

作者: 长弓路平
日期: 2024年10月22日
Retinal Image Dataset of Infants and ROP
"""

def get_configs():
    parser = argparse.ArgumentParser(description="Configuration for ROP dataset processing and model training")
    
    # 添加命令行参数
    parser.add_argument('--data_path', type=str, default='../Dataset/JTROP', help="Path to the dataset")
    parser.add_argument('--split_name', type=str, default='1', help="Name of the data split")
    parser.add_argument('--model_save_dir', type=str, default='../Model/MultiView', help="Directory to save the trained model")
    parser.add_argument('--experiments_dir', type=str, default='./experiments', help="Directory for experiment results")
    parser.add_argument('--model_config_path', type=str, default='./config_file/default.json', help="Path to the model configuration file")
    parser.add_argument('--dataset_name', type=str, default='JTROP', help="Name of the dataset")
    parser.add_argument('--un_enhanced', action='store_true', help="Use enhanced images")
    parser.add_argument('--save_model', action='store_true', help="Save the model")
    # training parameters
    parser.add_argument('--resize', type=int, default=224, help="Size to resize the images")
    parser.add_argument('--smoothing', type=float, default=0.1, help="Label smoothing")
    
    # 调整常见参数
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--wd', type=float, default=0.0001, help="Weight decay")
    # smoothing default = 0.1
    
    # 获取参数
    args = parser.parse_args()
    # 如果指定的目录不存在，则创建
    for dir_path in [args.model_save_dir, args.experiments_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # 加载模型配置文件
    if os.path.exists(args.model_config_path):
        with open(args.model_config_path, 'r') as f:
            args.model_config = json.load(f)
        print(f"Loaded model configuration from {args.model_config_path}")
    else:
        raise ValueError(f"Model configuration file {args.model_config_path} not found. Using default configuration.")

    return args

if __name__ == '__main__':
    args = get_configs()
    
    # 打印具体args的每一个参数
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

