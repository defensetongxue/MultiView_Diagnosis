import torch
from torch.utils.data import DataLoader
from util.dataset import CustomDataset
from  models import build_model
import os,json
import numpy as np
from util.metric import Metrics
from util.functions import train_epoch,val_epoch,get_optimizer,lr_sche
from configs import get_config
# Initialize random seed
torch.manual_seed(0)
np.random.seed(0)
from dataset_info import get_dbinfo
if __name__ == "__main__":
    # Parse arguments
    args = get_config()
    
    # load the config
    data_path=args.data_path
    db_info=get_dbinfo(args.dataset_name)
    num_classes=db_info.num_classes
    args.configs['model']['num_classes']=num_classes
    args.configs["lr_strategy"]["lr"]=args.lr
    args.configs['train']['lr']=args.lr
    args.configs['train']['wd']=args.wd
    enhanced=not args.un_enhanced
    
    # Create the model and criterion
    model= build_model(args.model_config["model"])# as we are loading the exite
    
    # Set up the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"using {device} for training")
    
    # early stopping
    early_stop_counter = 0
    
    # Creatr optimizer
    model.train()
    # Creatr optimizer
    optimizer = get_optimizer(args.model_config, model)
    lr_scheduler=lr_sche(config=args.model_config["lr_strategy"])
    last_epoch = args.model_config['train']['begin_epoch']
    
    # Load the datasets
    train_dataset=CustomDataset(
        split='train',
        data_path=data_path,
        split_name=args.split_name,
        resize=args.resize,
        enhanced=enhanced,bin=True)
    val_dataset=CustomDataset(
        split='val',
        data_path=data_path,
        split_name=args.split_name,
        resize=args.resize,
        enhanced=enhanced,bin=True)
    test_dataset=CustomDataset(
        split='test',
        data_path=data_path,
        split_name=args.split_name,
        resize=args.resize,
        enhanced=enhanced,bin=True)
    # Create the data loaders
        
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.configs['train']['batch_size'],
                              shuffle=True, num_workers=args.configs['num_works'],drop_last=True)
    test_loader=  DataLoader(test_dataset,
                            batch_size=args.configs['train']['batch_size'],
                            shuffle=False, num_workers=args.configs['num_works'])
    
    
        
    if args.configs['model']['name']=='inceptionv3':
        from models import incetionV3_loss
        assert args.resize>=299, "for the model inceptionv3, you should set resolusion at least 299 but now "
        
        criterion= incetionV3_loss(args.smoothing)
    else:
        if args.smoothing> 0.:
            from timm.loss import LabelSmoothingCrossEntropy
            criterion =LabelSmoothingCrossEntropy(args.smoothing)
            print("Using tmii official optimizier")
        else:
            # using defalut crss entropy
            criterion = torch.nn.CrossEntropyLoss()
            print("Using default CrossEntropyLoss")
            
    # init metic
    metirc= Metrics("Main",num_class=num_classes)
    
    print(f"Train: {len(train_loader)}, Test: {len(test_loader)}")
    
    early_stop_counter = 0
    best_val_loss = float('inf')
    best_auc=0
    best_avgrecall=0
    total_epoches=args.configs['train']['end_epoch']
    save_model_name=args.split_name+args.configs['save_name']
    saved_epoch=-1
    # Training and validation loop
    for epoch in range(last_epoch,total_epoches):
        train_loss = train_epoch(model, optimizer, train_loader, criterion, device,lr_scheduler,epoch)
        
        print(f"Epoch {epoch + 1}/{total_epoches}, "
          f"Train Loss: {train_loss:.6f}")
    if args.save_model:
        save_path = os.path.join(args.model_save_dir, f"{save_model_name}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
            
    # Evaluate the model on the test set
    metirc=Metrics("Main",num_class=num_classes)
    _, metirc=val_epoch(model, test_loader, criterion, device,metirc)
    print(f"Resukt:")
    print(metirc)
    param={
        "model":args.configs["model"]["name"],
        "args":args,
    }
    save_dir=os.path.join(args.experiments_dir,'result_record')
    os.makedirs(save_dir,exist_ok=True)
    # 获取时间 按照格式年份的后两位+月份+日期+小时+分钟 例如2411051902 表示2024年11月5日19点02分
    import time
    time_str=time.strftime("%y%m%d%H%M",time.localtime())
    metirc._store(param,save_path=os.path.join(save_dir,f"{time_str}.json"))