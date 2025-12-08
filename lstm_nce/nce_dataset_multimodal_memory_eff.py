import torch
import torch.nn as nn
from torch.utils.data import Dataset

import config_path, os
import pandas as pd
from prepare_dataset_vision_nce_multimodal_memory_eff import process_transcription, process_vision

class LSTM_NCE_Multimodal_DATASET(Dataset):
  def __init__(self, metadata, glove_model, stop_words_list, cache_limit=20):
    """
    metadata: list of dicts [{'participant_id': 300, 'group_count': 1, ...}, ...]
    glove_model: Pre-loaded Gensim KeyedVectors
    stop_word_list: list of stop words
    """
    super().__init__()
    self.metadata = metadata
    self.glove_model = glove_model
    self.stop_words_list = stop_words_list

    self.cache_limit = cache_limit
    self.cache_transcription = {} 
    self.cache_vision = {}

    self.cache_order = []

  def __len__(self):
    return len(self.metadata)
  
  def _manage_cache(self, p_id):
    if p_id not in self.cache_transcription:
      if len(self.cache_order) >= self.cache_limit:
        oldest_id = self.cache_order.pop(0)
        del self.cache_transcription[oldest_id]
        del self.cache_vision[oldest_id]
      
      self.cache_order.append(p_id)
  
  def _get_dataframe(self, p_id):
    if p_id not in self.cache_transcription:
      self._manage_cache(p_id)

      t_path = os.path.join(config_path.DATA_DIR, 'Transcription', f'{p_id}_transcript.csv')
      v_path = os.path.join(config_path.DATA_DIR, 'Vision Summary', f'{p_id}_vision_summary.csv')

      t_df = pd.read_csv(t_path)
      v_df = pd.read_csv(v_path)

      p_df, _ = process_transcription(t_df)
      v_processed = process_vision(v_df)

      self.cache_transcription[p_id] = p_df
      self.cache_vision[p_id] = v_processed

    return self.cache_transcription[p_id], self.cache_vision[p_id]
  
  def __getitem__(self, idx):
    row = self.metadata[idx]
    p_id = row['participant_id']
    count = row['group_count']
    start_time = row['start_time']
    stop_time = row['stop_time']

    participant_df, vision_df = self._get_dataframe(p_id)

    text_rows = participant_df.loc[participant_df['count'] == count]
    word_list = " ".join(text_rows.value.tolist()).split()

    trans_vectors = []
    for word in word_list:
      word = word.strip()
      if "'" in word: word = word.split("'")[0]
      # 필터링 로직
      if '<' in word or '>' in word or '_' in word or '[' in word or ']' in word or word in self.stop_words_list or word == "":
        pass
      elif word == "":
        pass
      else:
        try:
          trans_vectors.append(self.glove_model[word].tolist())
        except KeyError:
          pass

    v_target = vision_df.loc[(start_time <= vision_df.timestamp) & (vision_df.timestamp <= stop_time)]
    v_target = v_target.drop(columns=['timestamp'])
    vision_vectors = v_target.values.tolist()

    if len(trans_vectors) == 0:
      trans_vectors = [[0.0] * 300]
    if len(vision_vectors) == 0:
      vision_vectors = [[0.0] * (len(vision_df.columns)-1)]

    return torch.tensor(trans_vectors, dtype=torch.float32), torch.tensor(vision_vectors, dtype=torch.float32)