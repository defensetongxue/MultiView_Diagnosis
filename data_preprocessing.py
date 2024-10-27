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

    # 读取数据
    with open(os.path.join(data_path, 'annotations.json'), 'r') as f:
        data_dict = json.load(f)
    with open(os.path.join(data_path, 'follow_up.json'), 'r') as f:
        follow_up_dict = json.load(f)
    
    # 构建hid到image_names的映射，并生成hid_label
    hid_images = {}
    hid_label = {}
    
    for hid in follow_up_dict:
        image_names = []
        max_label = 0
        for record in follow_up_dict[hid]:
            for eye in ['OD', 'OS']:
                if eye in record['result']:
                    for image_name, label in record['result'][eye].items():
                        image_names.append(image_name)
                        max_label = max(max_label, label)
        hid_images[hid] = image_names
        hid_label[hid] = max_label
    
    # 初始化5个桶
    baket_dict = {i: [] for i in range(5)}
    
    # 根据hid_label预先划分数据集，每个hid_label随机分为5个桶，如果存在某一个hid_label的数据少于5个，则报错
    hid_label_type = set(hid_label.values())
    for label in hid_label_type:
        # 收集属于该label的hid
        hids = [hid for hid in hid_label if hid_label[hid] == label]
        # 如果某个label的hid数量少于5个，报错
        if len(hids) < 5:
            raise ValueError(f"Label {label} 的hid数量少于5个，无法进行5折划分")
        # 随机打乱
        random.shuffle(hids)
        # 划分桶
        for i, hid in enumerate(hids):
            baket_dict[i % 5].append(hid)
    
    # 生成5折交叉验证的数据集，每个都存储在一个json，3个用来train，1个用来val，1个用来test，每一个内部是image_name
    for fold in range(5):
        split_dict = {'train': [], 'val': [], 'test': []}
        test_basket = fold
        val_basket = (fold + 1) % 5
        train_baskets = [b for b in range(5) if b != test_basket and b != val_basket]
    
        # 将训练集数据放入split_dict['train']
        for b in train_baskets:
            for hid in baket_dict[b]:
                split_dict['train'].extend(hid_images[hid])
        # 将验证集数据放入split_dict['val']
        for hid in baket_dict[val_basket]:
            split_dict['val'].extend(hid_images[hid])
        # 将测试集数据放入split_dict['test']
        for hid in baket_dict[test_basket]:
            split_dict['test'].extend(hid_images[hid])
    
        # 保存到对应的json文件
        with open(f'dataset_fold_{fold}.json', 'w') as f:
            json.dump(split_dict, f)
    
        # 输出结果
        print(f"Fold {fold}:")
        print(f"Train set size: {len(split_dict['train'])}")
        print(f"Validation set size: {len(split_dict['val'])}")
        print(f"Test set size: {len(split_dict['test'])}")
        
    if GenEnhanced:
        os.makedirs(os.path.join(data_path,'enhanced'),exist_ok=True)
        enhancer=Enhancer()
        for image_name in data_dict:
            enhanced_path=os.path.join(data_path,'enhanced',data_dict[image_name]['image_name'])
            if os.path.exists(enhanced_path):
                if 'enhanced_path' not in data_dict[image_name]:
                    raise ValueError(f'enhanced_path not in data_dict in {image_name}')
                continue
            enhancer.enhanced_image(
                image_path=data_dict[image_name]['image_path'],
                save_path=enhanced_path)
            data_dict[image_name]['enhanced_path']=enhanced_path
    
        print('增强处理完成')
    
    # 生成mask 
    if GenMask:
        os.makedirs(os.path.join(data_path,'mask'),exist_ok=True)
        for image_name in data_dict:
            if os.path.exists(data_dict[image_name]['mask_path']):
                if 'mask_path' not in data_dict[image_name]:
                    raise ValueError(f'mask_path not in data_dict in {image_name}')
                continue
            mask_path=os.path.join(data_path,'mask',data_dict[image_name]['image_name'])
            generate_mask(
                image_path=data_dict[image_name]['image_path'],
                save_path=mask_path
            )
            data_dict[image_name]['mask_path']=mask_path
    
        print('掩码生成完成')
