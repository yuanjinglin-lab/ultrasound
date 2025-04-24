import os
import torch
import tqdm
import random
import copy
import numpy as np
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, recall_score, f1_score, precision_score, accuracy_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from dataset import MyDataset
from datetime import datetime
import json
import argparse
import itertools
from torchvision.utils import make_grid

# 导入您的UnifiedModel类
from models.unified_model import UnifiedModel

now = datetime.now()
time_str = now.strftime("%Y-%m-%d %H:%M:%S")

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化以保证实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def visualize_and_log(images, label, epoch, writer):
    def overlay_images(image, label):
        image_np = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        if label == 1:
            kind = 'dog'
        else:
            kind = 'cat'
        
        # 使用matplotlib绘制图像
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))
        ax.imshow(image_np)
        ax.set_title(kind)
        ax.axis('off')

        # 将图像转换为numpy数组
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plt.close(fig)
        return img[:,:,:3]

    batch_size = images.size(0)
    random_integers = [random.randint(0, batch_size-1) for _ in range(min(50, batch_size))]
    visualized_images = [overlay_images(images[i], label[i]) for i in random_integers]

    visualized_images = np.stack(visualized_images)
    visualized_images = torch.from_numpy(visualized_images).permute(0, 3, 1, 2)
    
    grid = make_grid(visualized_images, nrow=10)

    writer.add_image('Validation Images', grid, epoch)

def train_model(model, train_dataloader, validation_dataloader, optimizer, num_epochs, save_dir, exp_dir, config_name):
    writer = SummaryWriter(os.path.join(exp_dir, config_name))

    criterion = nn.CrossEntropyLoss()
    
    best_accuracy = 0
    best_f1 = 0
    
    for epoch in range(num_epochs):
        
        model.train()

        train_loss = 0
        validation_loss = 0
        validation_accuracy = 0
        
        train_bar = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch}/{num_epochs} - Training")
        
        for iteration, (data, label) in enumerate(train_bar):
            data = data.to(device)
            label = label.to(device)
            
            output = model(data)
            
            loss = criterion(output, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.data.item()
            
            train_bar.set_postfix({"Loss":  f"{train_loss / (iteration + 1):.4f}"})
            
        writer.add_scalar('Loss/train', train_loss / (iteration + 1), epoch)
        
        model.eval()  

        label_list = []
        predict_list = []
        scores_list = []

        val_bar = tqdm.tqdm(validation_dataloader, desc=f"Epoch {epoch}/{num_epochs} - Validation")
        
        for iteration, (data, label) in enumerate(val_bar):
            data, label = data.to(device), label.to(device)

            with torch.no_grad():
                
                output = model(data)
                loss = criterion(output, label)
                validation_loss += loss.item()
                
                preds = torch.argmax(output, dim=1).cpu().numpy()
                probs = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
                
                label_list.extend(label.cpu().numpy())
                predict_list.extend(preds)
                scores_list.extend(probs)                  

        accuracy = accuracy_score(label_list, predict_list)
        precision = precision_score(label_list, predict_list)
        recall = recall_score(label_list, predict_list)
        f1 = f1_score(label_list, predict_list)
        conf_matrix = confusion_matrix(label_list, predict_list)
        
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation Recall: {recall:.4f}")
        print(f"Validation F1: {f1:.4f}")
        print(f"Validation precision: {precision:.4f}")
        print(f"Validation Confusion Matrix: \n{conf_matrix}")

        writer.add_scalar('Loss/val', validation_loss / (iteration + 1), epoch)
        writer.add_scalar('Accuracy/val', accuracy, epoch)
        writer.add_scalar('Precision/val', precision, epoch)
        writer.add_scalar('Recall/val', recall, epoch)
        writer.add_scalar('F1/val', f1, epoch)
    
        fpr, tpr, thresholds = roc_curve(label_list, scores_list)
        roc_auc = auc(fpr, tpr)
        writer.add_scalar('AUC/val', roc_auc, epoch)
    
        # 按F1值保存最佳模型
        if f1 > best_f1:
            print(f"Saving best model for epoch {epoch + 1} (F1: {f1:.4f})")
            best_f1 = f1
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(save_dir, f"{config_name}_best.pth"))

            with open(os.path.join(save_dir, f"{config_name}_metrics.txt"), "w") as f:
                f.write(f"Accuracy: {accuracy:.4f}\n")
                f.write(f"Precision: {precision:.4f}\n")
                f.write(f"Recall: {recall:.4f}\n")
                f.write(f"F1 Score: {f1:.4f}\n")
                f.write(f"AUC: {roc_auc:.4f}\n")
                f.write("Confusion Matrix:\n")
                f.write(f"{conf_matrix}\n")

            labels_name = [str(i) for i in range(2)]  # 根据类别数调整
            plt.figure()
            sns.heatmap(conf_matrix.astype(int), annot=True, fmt='d', cmap='Blues',
                        square=True, xticklabels=labels_name, yticklabels=labels_name)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.savefig(os.path.join(save_dir, f"{config_name}_cm.png"), format='png')
            plt.close()

            # 绘制ROC曲线
            plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(save_dir, f"{config_name}_auc.png"))
            plt.close()
    
    return best_f1

def run_ablation_experiments(args, train_dataloader, validation_dataloader, fold):
    """运行一系列消融实验"""
    
    # 实验配置
    encoders = ["mobilenetv2", "resnet18", "rexnetv1", "swin"]
    fusion_methods = ["none", "concat", "tensor", "weighted", "cross"]
    pooling_options = ["max", "avg"]
    
    # 如果指定了单个配置，则只运行该配置
    if args.encoder and args.fusion and args.pooling:
        configs = [(args.encoder, args.fusion, args.pooling)]
    else:
        # 否则运行所有配置组合（完整消融实验）
        configs = list(itertools.product(encoders, fusion_methods, pooling_options))
    
    results = {}
    
    for encoder, fusion, pooling in configs:
        config_name = f"{encoder}_{fusion}_{pooling}"
        print(f"\n===== 运行配置: {config_name} =====")
        
        # 创建模型
        model = UnifiedModel(
            c_in=3,  # 假设输入是RGB图像
            num_classes=2,  # 二分类任务
            encoder_name=encoder,
            fusion_method=fusion,
            pooling=pooling
        ).to(device)
        
        # 创建优化器
        optimizer = optim.SGD(model.parameters(), args.lr, momentum=0.9, nesterov=True)
        
        # 为当前配置创建保存目录
        save_dir = Path(f"./check/fold_{fold}/{config_name}")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练模型
        best_f1 = train_model(
            model, 
            train_dataloader, 
            validation_dataloader, 
            optimizer,
            num_epochs=args.epochs, 
            save_dir=save_dir,
            exp_dir=args.exp_dir,
            config_name=config_name
        )
        
        # 保存结果
        results[config_name] = best_f1
    
    # 保存所有配置的结果
    with open(os.path.join(args.exp_dir, f"fold_{fold}_ablation_results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    return results

def summarize_ablation_results(results_all_folds, exp_dir):
    """汇总所有折叠的消融实验结果"""
    
    # 计算每个配置在所有折叠上的平均F1分数
    avg_results = {}
    
    for fold, results in results_all_folds.items():
        for config, f1 in results.items():
            if config not in avg_results:
                avg_results[config] = []
            avg_results[config].append(f1)
    
    # 计算平均值和标准差
    summary = {}
    for config, f1_scores in avg_results.items():
        summary[config] = {
            "mean_f1": np.mean(f1_scores),
            "std_f1": np.std(f1_scores),
            "f1_scores": f1_scores
        }
    
    # 按平均F1分数排序
    sorted_configs = sorted(summary.items(), key=lambda x: x[1]["mean_f1"], reverse=True)
    
    # 保存汇总结果
    with open(os.path.join(exp_dir, "ablation_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    
    # 生成表格和图表
    generate_summary_plots(sorted_configs, exp_dir)
    
    return sorted_configs

def generate_summary_plots(sorted_configs, exp_dir):
    """生成消融实验的汇总图表"""
    
    # 提取前5个配置用于绘图
    top_configs = sorted_configs[:5]
    
    # 柱状图：按编码器分组
    plt.figure(figsize=(12, 6))
    configs = [config[0] for config in top_configs]
    f1_means = [config[1]["mean_f1"] for config in top_configs]
    f1_stds = [config[1]["std_f1"] for config in top_configs]
    
    plt.bar(configs, f1_means, yerr=f1_stds, capsize=5)
    plt.xlabel('配置')
    plt.ylabel('F1分数')
    plt.title('消融实验结果：各配置的平均F1分数')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "ablation_f1_scores.png"))
    plt.close()
    
    # 分析不同编码器的性能
    encoder_perf = {}
    fusion_perf = {}
    pooling_perf = {}
    
    for config, metrics in sorted_configs:
        # 解析配置名
        encoder, fusion, pooling = config.split('_')
        
        # 按编码器聚合
        if encoder not in encoder_perf:
            encoder_perf[encoder] = []
        encoder_perf[encoder].append(metrics["mean_f1"])
        
        # 按融合方法聚合
        if fusion not in fusion_perf:
            fusion_perf[fusion] = []
        fusion_perf[fusion].append(metrics["mean_f1"])
        
        # 按池化方法聚合
        if pooling not in pooling_perf:
            pooling_perf[pooling] = []
        pooling_perf[pooling].append(metrics["mean_f1"])
    
    # 计算每个组件的平均性能
    encoder_avg = {k: np.mean(v) for k, v in encoder_perf.items()}
    fusion_avg = {k: np.mean(v) for k, v in fusion_perf.items()}
    pooling_avg = {k: np.mean(v) for k, v in pooling_perf.items()}
    
    # 创建组件性能图表
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # 编码器性能
    encoders = list(encoder_avg.keys())
    encoder_scores = list(encoder_avg.values())
    ax1.bar(encoders, encoder_scores)
    ax1.set_title('编码器平均性能')
    ax1.set_ylabel('平均F1分数')
    ax1.set_xticklabels(encoders, rotation=45, ha='right')
    
    # 融合方法性能
    fusions = list(fusion_avg.keys())
    fusion_scores = list(fusion_avg.values())
    ax2.bar(fusions, fusion_scores)
    ax2.set_title('融合方法平均性能')
    ax2.set_xticklabels(fusions, rotation=45, ha='right')
    
    # 池化方法性能
    poolings = list(pooling_avg.keys())
    pooling_scores = list(pooling_avg.values())
    ax3.bar(poolings, pooling_scores)
    ax3.set_title('池化方法平均性能')
    ax3.set_xticklabels(poolings, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "component_performance.png"))
    plt.close()
    
    # 生成汇总表格
    with open(os.path.join(exp_dir, "ablation_component_summary.txt"), "w") as f:
        f.write("编码器平均性能:\n")
        for encoder, score in sorted(encoder_avg.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{encoder}: {score:.4f}\n")
        
        f.write("\n融合方法平均性能:\n")
        for fusion, score in sorted(fusion_avg.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{fusion}: {score:.4f}\n")
        
        f.write("\n池化方法平均性能:\n")
        for pooling, score in sorted(pooling_avg.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{pooling}: {score:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练模型并进行消融实验')
    parser.add_argument('--epochs', type=int, default=100, help='训练的轮数')
    parser.add_argument('--lr', type=float, default=0.02, help='学习率')
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小')
    parser.add_argument('--seed', type=int, default=666, help='随机种子')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载的工作线程数')
    parser.add_argument('--encoder', type=str, default=None, help='特定编码器名称')
    parser.add_argument('--fusion', type=str, default=None, help='特定融合方法')
    parser.add_argument('--pooling', type=str, default=None, help='特定池化方法')
    parser.add_argument('--folds', type=int, default=5, help='交叉验证折数')
    
    args = parser.parse_args()
    
    # 设置实验目录
    args.exp_dir = Path(f"./exp/{time_str}")
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存参数配置
    with open(os.path.join(args.exp_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # 设置随机种子
    if args.seed is not None:
        seed_torch(args.seed)
    
    # 设置设备
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据集划分
    with open('split_result.json', 'r') as file:
        data_dict = json.load(file)
    
    # 记录所有折叠的结果
    all_fold_results = {}
    
    # 运行交叉验证
    for fold in range(args.folds):
        fold_data_dict = data_dict[f'fold_{fold}']
        print(f"\n===== Fold {fold} / {args.folds} =====")
        
        # 创建数据集
        train_dataset = MyDataset(fold_data_dict['train'], is_train=True)
        validation_set = MyDataset(fold_data_dict['validation'], is_train=False)

        # 创建数据加载器
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers, 
            drop_last=True,
            pin_memory=True
        )
        
        validation_dataloader = DataLoader(
            validation_set, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers, 
            drop_last=False,
            pin_memory=True
        )
        
        # 运行消融实验
        fold_results = run_ablation_experiments(args, train_dataloader, validation_dataloader, fold)
        all_fold_results[f"fold_{fold}"] = fold_results
    
    # 汇总所有折叠的结果
    best_configs = summarize_ablation_results(all_fold_results, args.exp_dir)
    
    # 打印最佳配置
    print("\n===== 消融实验汇总结果 =====")
    print("最佳配置 (按平均F1分数排序):")
    for i, (config, metrics) in enumerate(best_configs[:5]):
        print(f"{i+1}. {config}: 平均F1={metrics['mean_f1']:.4f} (±{metrics['std_f1']:.4f})")