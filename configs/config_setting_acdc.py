from .config_setting_synapse import setting_config as synapse_config
from utils import CeDiceLoss
from datetime import datetime

class setting_config(synapse_config):
    # ACDC有4个类别 (背景 + 3个器官)
    num_classes = 4
    data_path = './data'
    list_dir = './data'
    volume_path = './data' # 测试时使用h5文件
    z_spacing = 1
    is_pretrain = True
    
    # 保持Semantic-Aligned Skip开启
    use_sasf = True
    datasets_name = 'acdc'
    
    # 训练参数调整
    batch_size = 12
    max_epochs = 300
    base_lr = 0.0001
    val_interval = 10 # 每轮验证，方便调试
    
    # 更新model_config
    model_config = synapse_config.model_config.copy()
    model_config['num_classes'] = 4

    # ACDC结果单独保存到 results/acdc
    work_dir = f'results/acdc/{network}_{model_tag}_{datasets_name}_bs{batch_size}_{datetime.now().strftime("%Y%m%d_%H%M%S")}/'
    
    # 重新定义criterion以使用正确的num_classes
    criterion = CeDiceLoss(num_classes, synapse_config.loss_weight)
