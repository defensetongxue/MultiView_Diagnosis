"""
文件名: data_preprocessing.py

功能:
    1. 整理数据，生成annotations.json,给每个图片一个image_id方便处理
    2. 按照病人划分数据集成，5折交叉验证，划分为 0.6: 0.2: 0.2 存储在<number>.json 中
    3. 生成mask和enhanced

备注：使用images_stack_without_captions文件夹并把2级的images_stack_without_captions重命名为了images
作者: 长弓路平
日期: 2024年10月22日
"""
import os,json,random
from util.pre_process_tool import Enhancer,generate_mask
from configs import get_configs
from util.tools import unpack_name
from collections import defaultdict
GenEnhanced=True
GenMask=False

if __name__=='__main__':
    args=get_configs()
    
    data_dict={}
    data_path= args.data_path
    
    print(f'处理数据集路径 {data_path}')
    #  每一行都是一个数据,赋予一个image_id
    image_name_list=sorted(os.listdir(os.path.join(data_path,'images')))
    
    # 初始化5个桶
    baket_dict = {i: [] for i in range(1, 6)}
    patient_buckets = defaultdict(list) 

    # 记录所有的患者ID
    patient_set = set() 

    # 将图片根据患者ID进行分组
    for image_id, image_name in enumerate(image_name_list):
        data_info = unpack_name(image_name)
        image_path = os.path.join(data_path, 'images', image_name)
        data_dict[image_id] = {
            **data_info,  # 解构data_info
            'image_path': image_path,
            'image_id': image_id,
            'image_name': image_name
        }
        patient_set.add(data_info['patient_id'])    

    # 将患者ID排序，并shuffle
    patient_list = sorted(patient_set)
    random.shuffle(patient_list)  # 在分配前shuffle，增加随机性
    num_patients = len(patient_list)
    split_indices = [
        int(0.2 * num_patients),
        int(0.4 * num_patients),
        int(0.6 * num_patients),
        int(0.8 * num_patients),
        num_patients
    ]   

    # 将患者分配到相应的桶
    for i, patient_id in enumerate(patient_list):
        if i < split_indices[0]:
            bucket_id = 1
        elif i < split_indices[1]:
            bucket_id = 2
        elif i < split_indices[2]:
            bucket_id = 3
        elif i < split_indices[3]:
            bucket_id = 4
        else:
            bucket_id = 5   

        # 将该患者的所有图像放入对应的桶
        for image_id, data in data_dict.items():
            if data['patient_id'] == patient_id:
                baket_dict[bucket_id].append(image_id)  

    # 按照比例分配 train/val/test
    for split_name in range(1, 6):
        split_list = {'train': [], 'val': [], 'test': []}  # 都是id_list
        for bucket_id, image_ids in baket_dict.items():
            total_images = len(image_ids)
            train_split = int(0.7 * total_images)
            val_split = int(0.85 * total_images)    

            # 将每个桶中的数据按比例分为train, val, test
            split_list['train'].extend(image_ids[:train_split])
            split_list['val'].extend(image_ids[train_split:val_split])
            split_list['test'].extend(image_ids[val_split:])    

        # 保存分割结果
        os.makedirs(os.path.join(data_path, 'split'), exist_ok=True)
        split_file_path = os.path.join(data_path, 'split', f'{split_name}.json')
        json.dump(split_list, open(split_file_path, 'w'), indent=4) 

        print(f'Split {split_name} saved to {split_file_path}')
    
    # 生成enhanced
    if GenEnhanced:
        os.makedirs(os.path.join(data_path,'enhanced'),exist_ok=True)
        enhancer=Enhancer()
        for image_id in data_dict:
            enhanced_path=os.path.join(data_path,'enhanced',data_dict[image_id]['image_name'])
            enhancer.enhanced_image(
                image_path=data_dict[image_id]['image_path'],
                save_path=enhanced_path)
            data_dict[image_id]['enhanced_path']=enhanced_path
    
        print('增强处理完成')
    
    # 生成mask 
    if GenMask:
        os.makedirs(os.path.join(data_path,'mask'),exist_ok=True)
        for image_id in data_dict:
            mask_path=os.path.join(data_path,'mask',data_dict[image_id]['image_name'])
            generate_mask(
                image_path=data_dict[image_id]['image_path'],
                save_path=mask_path
            )
            data_dict[image_id]['mask_path']=mask_path
    
        print('掩码生成完成')
