import os, config_path, sys, json, yaml, argparse, gc
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from loguru import logger

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast, GradScaler

from gensim.models import KeyedVectors

import nltk
from nltk.corpus import stopwords

from lstm_nce_multimodal import NCEMultimodalModel
from prepare_dataset_vision_nce_multimodal_memory_eff import prepare_metadata
from nce_dataset_multimodal_memory_eff import LSTM_NCE_Multimodal_DATASET

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)

logger.remove()
logger.add(
  sys.stdout,
  colorize=True,
  format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)

def collate_batch(batch):
  # batch: list of (tensor(tokens), tensor(label))
  x = [item[0] for item in batch]
  x2 = [item[1] for item in batch]
  x_lengths = torch.tensor([t.size(0) for t in x], dtype=torch.long)
  x2_lengths = torch.tensor([t2.size(0) for t2 in x2], dtype=torch.long)
  x_padded = pad_sequence(x, batch_first=True, padding_value=0) # pad id = 0
  x2_padded = pad_sequence(x2, batch_first=True, padding_value=0) # pad id = 0
  return x_padded, x_lengths, x2_padded, x2_lengths

@torch.no_grad()
def update_momentum_encoder(encoder_k, encoder_q, m=0.99):
  for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
    param_k.data = param_k.data * m + param_q.data * (1. - m)

class Queue(nn.Module):
  def __init__(self, output_dim:int, queue_size=4096):
    super().__init__()
    self.register_buffer("queue", torch.randn(output_dim, queue_size))
    self.queue = F.normalize(self.queue, dim=0)
    self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    self.register_buffer("is_initialized", torch.tensor(False))

  @torch.no_grad()
  def initialize_queue(self, keys):
    if not self.is_initialized: # type: ignore
      batch_size = keys.shape[0]
      # repeat with first batch
      for i in range(0, self.queue.shape[1], batch_size):
        end = min(i + batch_size, self.queue.shape[1])
        self.queue[:, i:end] = keys[:end-i].T
      self.is_initialized.fill_(True) # type: ignore
  
  @torch.no_grad()
  def dequeue_and_enqueue(self, keys):
    keys = keys.detach()
    batch_size = keys.shape[0]
    ptr = int(self.queue_ptr) # type: ignore
    queue_size = self.queue.shape[1]

    end = (ptr + batch_size) % queue_size
    
    if end > ptr:
      self.queue[:, ptr:end] = keys[:end - ptr].T
    else:
      remaining = queue_size - ptr
      self.queue[:, ptr:] = keys[:remaining].T
      self.queue[:, :end] = keys[remaining:].T

    self.queue_ptr[0] = end # type: ignore

def train(model, dataloader, x_queue, x2_queue, optimizer, device, config, criterion, lmbda, warmup=False, temperature=0.07, is_sample=False):
  model.encoder_x_q.train()
  model.encoder_x2_q.train()
  model.encoder_x_k.eval()
  model.encoder_x2_k.eval()

  scaler = GradScaler()

  total_loss = 0.0
  total_x_std = 0.0
  total_x2_std = 0.0
  num_batches = len(dataloader)
  pbar = tqdm(dataloader, desc='Training', ncols=120)

  for batch_idx, (x, x_len, x2, x2_len) in enumerate(pbar):
    # x = transcription data, x2 = different modality data
    x, x_len, x2, x2_len = x.to(device), x_len.to(device), x2.to(device), x2_len.to(device)

    if torch.isnan(x).any() or torch.isinf(x).any():
      logger.info("입력 데이터 X에 NaN/Inf 발견!")

    if warmup:
      with torch.no_grad():
        k_feat, _ = model.encoder_x_k(x, x_len)
        k = F.normalize(k_feat, dim=1)
        x_queue.dequeue_and_enqueue(k)
        k2_feat, _ = model.encoder_x2_k(x2, x2_len)
        k2 = F.normalize(k2_feat, dim=1)
        x2_queue.dequeue_and_enqueue(k2)
      continue

    optimizer.zero_grad(set_to_none=True)
    with autocast():
      result1, result2, k, q_std = model(x, x_len, x2, x2_len, x_queue.queue.clone().detach(), x2_queue.queue.clone().detach(), is_sample)
      loss1 = criterion(result1[0]/temperature, result1[1])
      loss2 = criterion(result2[0]/temperature, result2[1])
      final_loss = loss1 + loss2
    
    scaler.scale(final_loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        list(model.encoder_x_q.parameters()) + list(model.encoder_x2_q.parameters()), 
        max_norm=5.0
    )
    scaler.step(optimizer)
    scaler.update()

    # compute key features (momentum encoder)
    with torch.no_grad():
      update_momentum_encoder(model.encoder_x_k, model.encoder_x_q, config['training']['M'])
      update_momentum_encoder(model.encoder_x2_k, model.encoder_x2_q, config['training']['M'])

    with torch.no_grad():
      x_queue.dequeue_and_enqueue(k[0])
      x2_queue.dequeue_and_enqueue(k[1])

    total_loss += final_loss.item()
    total_x_std += q_std[0].item()
    total_x2_std += q_std[1].item()

    avg_loss = total_loss / (batch_idx + 1)
    avg_x_std = total_x_std / (batch_idx + 1)
    avg_x2_std = total_x2_std / (batch_idx + 1)

    # current_lr = optimizer.param_groups[0]['lr']

    total_norm = 0
    for p in model.encoder_x_q.parameters():
      if p.grad is not None:
        total_norm += p.grad.norm(2).item() ** 2
    for p in model.encoder_x2_q.parameters():
      if p.grad is not None:
        total_norm += p.grad.norm(2).item() ** 2

    # Update progress bar with detailed info
    pbar.set_postfix({
      'avg loss': f'{avg_loss:.4f}',
      'grad norm': f'{total_norm**0.5:.4f}',
      'x std': f'{avg_x_std:.4f}',
      'x2 std': f'{avg_x2_std:.4f}',
      'batch': f'{batch_idx+1}/{num_batches}'
    })

  if warmup:
    return 0.0, 0.0, 0.0
  else:
    return avg_loss, avg_x_std, avg_x2_std
"""
def validation_knn(model, train_loader, val_loader, device, criterion, lmbda, k=5, batch_size=256):
  model.encoder_k.eval()
  
  print("Building feature bank from training set...")
  train_features = []
  with torch.no_grad():
    for x, x_len, x2, x2_len in tqdm(train_loader, desc='Feature Bank', ncols=100):
      x, x_len, x2, x2_len = x.to(device), x_len.to(device), x2.to(device), x2_len.to(device)
      k_feat, _ = model.encoder_k(x2, x2_len)
      features = F.normalize(k_feat, dim=1)
      train_features.append(features.cpu())  # CPU로 이동하여 메모리 절약
  
  train_features = torch.cat(train_features, dim=0)

  print(f"Running KNN validation (k={k})...")
  correct = 0
  total = 0
  num_batches = len(val_loader)

  with torch.no_grad():
    pbar = tqdm(val_loader, desc='KNN Validation', ncols=120)
    for batch_idx, (x, x_len, x2, x2_len) in enumerate(pbar):
      x, x_len, x2, x2_len = x.to(device), x_len.to(device), x2.to(device), x2_len.to(device)
      
      k_feat, _ = model.encoder_k(x2, x2_len)
      features = F.normalize(k_feat, dim=1)
      
      batch_predictions = []
      for i in range(0, train_features.size(0), batch_size):
        batch_train_features = train_features[i:i+batch_size].to(device)
        batch_sim = torch.matmul(features, batch_train_features.T)
        batch_predictions.append(batch_sim)
      
      similarities = torch.cat(batch_predictions, dim=1)
      
      # Top-k
      _, topk_indices = similarities.topk(k, dim=1)
      topk_labels = train_labels[topk_indices.cpu()]
      predictions = torch.mode(topk_labels, dim=1)[0]
      
      correct += (predictions.to(device) == y).sum().item()
      total += y.size(0)
      
      pbar.set_postfix({
          'acc': f'{correct/total:.4f}',
          'correct_knn': f'{correct}/{total}',
          'batch': f'{batch_idx+1}/{num_batches}'
      })
  accuracy = correct / total
  print(f"\nKNN Validation Accuracy (k={k}): {accuracy:.4f}")
  return accuracy
"""
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--num_epochs', default=100, type=int,
                      help='Number of training epochs.')
  parser.add_argument('--resume', default='', type=str, metavar='PATH',
                      help='Path to latest checkpoint (default: none).')
  parser.add_argument('--config', type=str, default='lstm_nce/configs/architecture.yaml',
                      help="Which configuration to use. See into 'config' folder.")
  parser.add_argument('--save_dir', type=str, default='checkpoints', metavar='PATH',
                      help="Directory path to save model")
  parser.add_argument('--save_dir_', type=str, default='checkpoints1', metavar='PATH',
                      help="Exact directory path to save model")
  parser.add_argument('--patience', type=int, default=10, 
                      help="How many epochs wait before stopping for validation loss not improving.")
  parser.add_argument('--warmup_epochs', type=int, default=3,
                      help='Number of warm-up epochs to fill the queue before full training.')
  parser.add_argument('--is_sample', type=bool, default=False,
                      help='Used for checking model outputs')
  parser.add_argument('--cache_limit', type=int, default=20,
                      help='How many dataframes to cache')
  
  opt = parser.parse_args()
  if opt.is_sample:
    opt.num_epochs = 2
    opt.warmup_epochs = 0
    logger.info("Sample -> Changed to epochs 2 & warmup 0")
  logger.info(opt)

  with open(os.path.join(config_path.ROOT_DIR, opt.config), 'r', encoding="utf-8") as ymlfile:
    config = yaml.safe_load(ymlfile)

  os.makedirs(opt.save_dir, exist_ok=True)
  CHECKPOINTS_DIR = os.path.join(opt.save_dir, opt.save_dir_)
  os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

  history = {
    'epoch':[],
    'train_loss':[],
    'train_x_std':[],
    'train_x2_std':[]
    # 'train_correct':[],
    # 'val_accuracy':[]
  }

  if opt.is_sample:
    logger.info("Sample Data: 300/301")
    train_id = [300]
    val_id = [301]
  
  else:
    train_df = pd.read_csv(os.path.join(config_path.DATA_DIR, 'train_split_Depression_AVEC2017.csv'))
    val_df = pd.read_csv(os.path.join(config_path.DATA_DIR, 'dev_split_Depression_AVEC2017.csv'))
    
    train_id = train_df.Participant_ID.tolist()
    val_id = val_df.Participant_ID.tolist()

  logger.info("Loading GLOVE")
  glove_kv_path = os.path.join(config_path.MODEL_DIR, "glove_model.kv")

  assert os.path.exists(glove_kv_path), "No GLOVE Model"

  # Defining GLOVE
  try:
    glove_model = KeyedVectors.load(glove_kv_path)
    logger.info("Loaded GLOVE")
  except Exception as e:
    logger.error(f"Problem with your GLOVE: {e}")

  # Defining STOPWORDS
  nltk.download('stopwords')
  stop_words_list = stopwords.words('english')
  logger.info("Loaded Stopwords")

  logger.info("Preparing Train Metadata indices...")
  train_meta = prepare_metadata(train_id+val_id, stop_words_list, glove_model)
    
  logger.info(f"Train Samples: {len(train_meta)}")

  logger.info("Dataset Ready")

  train_data = LSTM_NCE_Multimodal_DATASET(train_meta, glove_model, stop_words_list, cache_limit=opt.cache_limit)
  sample_x, sample_x2 = train_data[0]
  x_input_dim = sample_x.shape[1]
  x2_input_dim = sample_x2.shape[1]
  # train_vision_dataset: (id*B, seq_len, v_columns)
  logger.info(f"Current Transcription Dimenseion: {x_input_dim}")
  logger.info(f"Current Multimodal Dimenseion: {x2_input_dim}")

  del train_meta, glove_model, stop_words_list
  gc.collect()
  torch.cuda.empty_cache()

  logger.info("DataLoader Ready")
  if opt.is_sample:
    train_loader = DataLoader(train_data, batch_size=1, num_workers=0, shuffle=False, collate_fn=collate_batch)
  else:
    train_loader = DataLoader(train_data, batch_size=config['training']['bs'], num_workers=config['training']['workers'], shuffle=True, pin_memory=True, collate_fn=collate_batch)

  # train setting
  hidden_dim = config['model']['h_dim']
  output_dim = config['model']['o_dim']
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  model = NCEMultimodalModel(x_input_dim, x2_input_dim, hidden_dim, output_dim, num_layers=config['model']['depth'], dropout=config['model']['dropout'])
  if opt.is_sample:
    queue_x = Queue(output_dim=output_dim, queue_size=2)
    queue_x2 = Queue(output_dim=output_dim, queue_size=2)
  else:
    queue_x = Queue(output_dim=output_dim, queue_size=config['training']['q_size'])
    queue_x2 = Queue(output_dim=output_dim, queue_size=config['training']['q_size'])
  criterion = CrossEntropyLoss()
  optimizer = torch.optim.AdamW(
    list(model.encoder_x_q.parameters()) + list(model.encoder_x2_q.parameters()),
    lr=float(config['training']['lr']),
    weight_decay=float(config['training']['weight-decay'])
    )
  if config['training']['scheduler'] == 'steplr':
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config['training']['step-size'], gamma=config['training']['gamma'])
  else:
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.num_epochs)

  model.to(device)
  queue_x.to(device)
  queue_x2.to(device)

  logger.info("Model/Queue Ready")

  starting_epoch = 0
  if opt.resume and os.path.exists(opt.resume):
    logger.info(f"Loading checkpoint from {opt.resume}")
    checkpoint = torch.load(opt.resume)

    try:
      model.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
      queue_x.load_state_dict(checkpoint['queue_x_state_dict'])
      queue_x2.load_state_dict(checkpoint['queue_x2_state_dict'])
      starting_epoch = checkpoint['epoch'] + 1
      history = checkpoint['history']
    except Exception as e:
      logger.error("Error Resuming:", e)
      logger.error("Check your configuration files.")
      return
    
    del checkpoint
    gc.collect()
    logger.info(f"Resuming from epoch {starting_epoch}")
    logger.info(f"Previous best val_loss: {min(history['val_loss']) if history['val_loss'] else 'N/A'}")
  else:
    logger.info("No checkpoint loaded. Starting from scratch.")

  logger.info("Train Start")

  patience = 0

  for epoch in range(starting_epoch, opt.num_epochs + 1 + opt.warmup_epochs):
    warmup = epoch <= opt.warmup_epochs
    if warmup:
      logger.info(f"\n[Warm-up Epoch {epoch}/{opt.warmup_epochs}] Filling queue only...")
    if opt.is_sample:
      _, _, _ =\
        train(model=model,
              dataloader=train_loader,
              x_queue=queue_x,
              x2_queue=queue_x2,
              optimizer=optimizer,
              device=device,
              config=config,
              criterion=criterion,
              lmbda=config['training']['lambda'],
              warmup=warmup,
              temperature=config['training']['T'],
              is_sample=True)
    else:
      train_avg_loss, train_avg_x_std, train_avg_x2_std =\
        train(model=model,
              dataloader=train_loader,
              x_queue=queue_x,
              x2_queue=queue_x2,
              optimizer=optimizer,
              device=device,
              config=config,
              criterion=criterion,
              lmbda=config['training']['lambda'],
              warmup=warmup,
              temperature=config['training']['T'])  
    if warmup:
      continue
    # val_accuracy = validation_knn(model=model, train_loader=train_loader, val_loader=val_loader, device=device, lmbda=config['training']['lambda'], temperature=config['training']['T'])

    history['epoch'].append(epoch)
    history['train_loss'].append(train_avg_loss)
    history['train_x_std'].append(train_avg_x_std)
    history['train_x2_std'].append(train_avg_x2_std)
    # history['train_correct'].append(total_correct_rate)
    # history['val_accuracy'].append(val_accuracy)

    logger.info("#" + str(epoch) + "/" + str(opt.num_epochs) + " loss:" +
                str(train_avg_loss)) # + " accuracy:" + str(val_accuracy)
    
    checkpoint = {
        'epoch': epoch-opt.warmup_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'queue_x_state_dict' : queue_x.state_dict(),
        'queue_x2_state_dict' : queue_x2.state_dict(),
        'config' : config,
        'history': history
    }

    if epoch % 5 == 0:
      checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"checkpoint_epoch{epoch}.pth")
      torch.save(checkpoint, checkpoint_path)
      logger.info(f"Checkpoint saved: {checkpoint_path}")

    if len(history['train_loss']) > 0 and train_avg_loss <= min(history['train_loss']):
      best_path = os.path.join(CHECKPOINTS_DIR, f"best_model.pth")
      torch.save(checkpoint, best_path)
      logger.info(f"Best model saved: {best_path}")
      patience = 0
    else:
      patience += 1
      logger.info(f"Current Patience: {patience}")
    
    if patience >= opt.patience:
      logger.info(f"# {epoch}/{opt.num_epochs} => Early Stopping")
      break

  # 학습 완료 후 그래프 생성
  logger.info("\nGenerating training history plots...")

  # 디렉토리 생성
  plots_dir = os.path.join(CHECKPOINTS_DIR, "plots")
  if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

  epochs_range = range(starting_epoch, starting_epoch + len(history['train_loss']))

  fig, axes = plt.subplots(1, 3, figsize=(18, 5))

  # 1. Loss 그래프 (첫 번째 칸: axes[0])
  axes[0].plot(epochs_range, history['train_loss'], 'b-', label='Train Loss')
  axes[0].set_xlabel('Epoch')
  axes[0].set_ylabel('Loss')
  axes[0].set_title('Training Loss')
  axes[0].legend()
  axes[0].grid(True)

  # 2. X_std 그래프 (두 번째 칸: axes[1])
  axes[1].plot(epochs_range, history['train_x_std'], 'g-', label='Train X Std') # 색상 구분(green)
  axes[1].set_xlabel('Epoch')
  axes[1].set_ylabel('Std')
  axes[1].set_title('Training X Std')
  axes[1].legend()
  axes[1].grid(True)

  # 3. X2_std 그래프 (세 번째 칸: axes[2])
  axes[2].plot(epochs_range, history['train_x2_std'], 'r-', label='Train X2 Std') # 색상 구분(red)
  axes[2].set_xlabel('Epoch')
  axes[2].set_ylabel('Std')
  axes[2].set_title('Training X2 Std')
  axes[2].legend()
  axes[2].grid(True)

  plt.tight_layout()
  
  # 이미지 저장
  plot_path = os.path.join(plots_dir, 'training_history.png')
  plt.savefig(plot_path, dpi=300)
  
  # 메모리 해제 (중요: 반복문 등에서 plt가 쌓이는 것 방지)
  plt.close(fig) 
  
  logger.info(f"Plot saved: {plot_path}")

  # 히스토리를 JSON으로 저장
  history_path = os.path.join(CHECKPOINTS_DIR, f'training_history.json')
  with open(history_path, 'w') as f:
      json.dump(history, f, indent=4)
  logger.info(f"History saved: {history_path}")

  logger.info("\nTraining completed!")
  logger.info(f"Best Train loss: {min(history['train_loss']):.4f}")

if __name__=="__main__":
  main()

# first
#   ex) python lstm_nce/train_nce_multimodal_memory_eff.py --save_dir checkpoints_nce --save_dir_ multimodal_nce_1 --warmup_epochs 1
#     -> epochs: 100, config_file: lstm_nce/configs/architecture.yaml, save_path: checkpoints/checkpoints1, patience: 10, warmup_for_queue: 3
# resume
#   ex) python lstm_nce/train_nce.py --resume checkpoints/checkpoints1/best_model.pth