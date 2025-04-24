import json
from pathlib import Path
from sklearn.model_selection import KFold

# 参数分别为pkl地址、json文件保存地址、折数
def main(root_path,save_path,fold=5):
    
    data_path = Path(root_path)
    all_data_list = list(data_path.glob('*.npy'))

    kf = KFold(n_splits=fold, shuffle=True, random_state=666)

    split_data = {}
    for fold_index, (train_index, validation_index) in enumerate(kf.split(all_data_list)):
        train_set = [str(all_data_list[i].absolute()) for i in train_index]
        validation_set = [str(all_data_list[i].absolute()) for i in validation_index]
        
        dict_name = f'fold_{fold_index}'
        split_data[dict_name] = {'train': train_set,'validation': validation_set}
        
    with open(save_path, 'w') as file:
        json.dump(split_data, file, indent=4)
    print("Data has been successfully split and written .")

if __name__ == '__main__':
    # 创建多折训练划分json文件
    main('/home/wangnannan/workdir/ultrasound/data/numpy_data','split_result.json')
    # main('/home/wangnannan/workdir/5_3d_cls-2-master/data','split_result.json')