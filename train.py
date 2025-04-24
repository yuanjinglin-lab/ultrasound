import os
import torch
import tqdm
import random
import copy
import numpy as np
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, recall_score, f1_score, precision_score,accuracy_score,roc_auc_score,roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from model import Model
from torchvision.utils import make_grid
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from dataset import MyDataset,dog_and_cat
from datetime import datetime

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

    def overlay_images(image,label):
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
    random_integers = [random.randint(0, batch_size-1) for _ in range(50)]
    visualized_images = [overlay_images(images[i], label[i]) for i in random_integers]

    visualized_images = np.stack(visualized_images)
    visualized_images = torch.from_numpy(visualized_images).permute(0, 3, 1, 2)
    
    grid = make_grid(visualized_images, nrow=10)

    writer.add_image('Validation Images', grid, epoch)

def train_model(model, train_dataloader, validation_dataloader, optimizer,  num_epochs, save_dir,exp_dir):
    writer = SummaryWriter(exp_dir)

    criterion = nn.CrossEntropyLoss()
    
    best_accuracy = 0
    
    for epoch in range(num_epochs):
        
        model.train()

        train_loss = 0
        validation_loss = 0
        validation_accuracy = 0
        
        train_bar = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch}/{num_epochs} - Training")
        
        for iteration,(data, label) in enumerate(train_bar):
            data = data.to(device)
            label = label.to(device)
            
            output = model(data)
            
            loss = criterion(output, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.data.item()
            
            train_bar.set_postfix({"Loss":  f"{train_loss / (iteration + 1):.4f}"})
            
        writer.add_scalar('Loss/train',train_loss / (iteration + 1),epoch)
        
        model.eval()  

        label_list = []
        predict_list = []
        scores_list = []

        val_bar = tqdm.tqdm(validation_dataloader, desc=f"Epoch {epoch}/{num_epochs} - Validation")
        
        for iteration, (data, label) in enumerate(val_bar):
            data,label = data.to(device),label.to(device)

            with torch.no_grad():
                
                output = model(data)
                loss = criterion(output, label)
                validation_loss += loss.item()
                
                preds = torch.argmax(output, dim=1).cpu().numpy()
                probs = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
                
                label_list.extend(label.cpu().numpy())
                predict_list.extend(preds)
                scores_list.extend(probs)                  

            # val_bar.set_postfix({"Loss": f"{validation_loss / (iteration + 1) :.2f}","ACC": f"{validation_accuracy / (iteration + 1):.2f}"})

        accuracy = accuracy_score(label_list, predict_list)
        precision = precision_score(label_list, predict_list)
        recall = recall_score(label_list, predict_list)
        f1 = f1_score(label_list, predict_list)
        conf_matrix = confusion_matrix(label_list, predict_list)
        
        print(f"Validation Accuracy: {accuracy}")
        print(f"Validation Recall: {recall}")
        print(f"Validation Confusion Matrix: \n{conf_matrix}")
        print(f"Validation F1: {f1}")
        print(f"Validation precision: {precision}")

        val_acc = validation_accuracy / (iteration + 1)
        writer.add_scalar('Loss/val',validation_loss / (iteration + 1),epoch)
        writer.add_scalar('Accuracy/val',val_acc,epoch)

        fpr, tpr, thresholds = roc_curve(label_list, scores_list)
        roc_auc = auc(fpr, tpr)
    
        if val_acc > best_accuracy:
            print(f"Saving best model for epoch {epoch + 1}")
            best_accuracy = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(save_dir, "epoch_best.pth"))

            with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
                f.write(f"Accuracy: {val_acc:.4f}\n")
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
            plt.savefig(os.path.join(save_dir, "cm.png"), format='png')
            plt.close()

            # 绘制ROC曲线
            plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange',lw=lw, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(save_dir, "auc.png"))
            plt.close()
            
if __name__ == "__main__":
    # seed_torch(666)
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    with open('split_result.json', 'r') as file:
        data_dict = json.load(file)
    
    n_splits = 5

    for fold in range(n_splits):
        
        fold_data_dict = data_dict[f'fold_{fold}']
        print(f"\n===== Fold {fold} / {n_splits} =====")
        
        train_dataset = MyDataset(fold_data_dict['train'], is_train=True)
        validation_set = MyDataset(fold_data_dict['validation'], is_train=False)

        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True,pin_memory=True)
        validation_dataloader = DataLoader(validation_set, batch_size=1, shuffle=False, num_workers=0, drop_last=False,pin_memory=True)
        
        model = Model(c_in=3, num_classes=2).to(device)

        optimizer = optim.SGD(model.parameters(), 0.02, momentum = 0.9, nesterov=True)

        save_dir = Path(f"./check/fold_{fold}")
        save_dir.mkdir(parents=True,exist_ok=True)
        
        exp_dir = Path(f"./exp/") / time_str
        exp_dir.mkdir(parents=True,exist_ok=True)
        
        train_model(model, train_dataloader, validation_dataloader, optimizer,num_epochs=100, save_dir=save_dir,exp_dir=exp_dir)