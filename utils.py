def download_weights(backbone, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url
    
    download_urls = {
        'mobilenetv2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
        'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
        'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
        'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
        'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
        'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
        'vit_b_16': 'https://github.com/bubbliiiing/classification-pytorch/releases/download/v1.0/vit-patch_16.pth',
        'swin_transformer_tiny': 'https://github.com/bubbliiiing/classification-pytorch/releases/download/v1.0/swin_tiny_patch4_window7_224_imagenet1k.pth',
        'swin_transformer_small': 'https://github.com/bubbliiiing/classification-pytorch/releases/download/v1.0/swin_small_patch4_window7_224_imagenet1k.pth',
        'swin_transformer_base': 'https://github.com/bubbliiiing/classification-pytorch/releases/download/v1.0/swin_base_patch4_window7_224_imagenet1k.pth'
    }
    try:
        url = download_urls[backbone]
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        load_state_dict_from_url(url, model_dir)
    except:
        print("There is no pretrained model for " + backbone)