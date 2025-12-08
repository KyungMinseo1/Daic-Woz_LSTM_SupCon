import os, config_path, sys
from loguru import logger
import pandas as pd
from tqdm import tqdm
from gensim.models import KeyedVectors

import nltk
from nltk.corpus import stopwords

logger.remove()
logger.add(
  sys.stdout,
  colorize=True,
  format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)

def process_transcription(df):
  finish_utterance = ["asked everything", "asked_everything", "it was great chatting with you"]

  search_pattern = '|'.join(finish_utterance)
  condition = df['value'].str.contains(search_pattern, na=False)
  terminate_index = df.index[condition]
  if terminate_index.empty:
    terminate_value = len(df)
  else:
    terminate_value = terminate_index.values[0]
  n_df = df.iloc[:terminate_value].copy()

  is_not_ellie = n_df['speaker'] != 'Ellie'
  new_group_start = (is_not_ellie) & (~is_not_ellie.shift(1, fill_value=False))
  group_id = new_group_start.cumsum()

  n_df['count'] = group_id.where(is_not_ellie, pd.NA)
  participant_df = n_df.dropna(subset=['count'])
  group_df = participant_df.dropna(subset=['count']).groupby('count').agg(
      start_time=('start_time', 'min'), # 그룹의 가장 이른 시작 시간
      stop_time=('stop_time', 'max')   # 그룹의 가장 늦은 종료 시간
  ).reset_index()
  return participant_df, group_df

def process_vision(df):
  timestamp = df.timestamp
  ft_x = df.filter(like='ftx')
  ft_y = df.filter(like='fty')
  # ft_3d_x = df.filter(like='ft_3dX')
  # ft_3d_y = df.filter(like='ft_3dY')
  # ft_3d_z = df.filter(like='ft_3dZ')
  au_r = df.filter(like='au').filter(like='_r')
  gz_df = df.filter(like='gz')
  gz_h = gz_df.filter(like='h')
  ps_t = df.filter(like='ps').filter(like='T')
  ps_r = df.filter(like='ps').filter(like='R')
  vision = pd.concat([timestamp, ft_x, ft_y, au_r, gz_h, ps_t, ps_r], axis=1) # pd.concat([timestamp, ft_x, ft_y, ft_3d_x, ft_3d_y, ft_3d_z, au_r, gz_h, ps_t, ps_r], axis=1)
  return vision

def get_valid_indices(id, stop_words_list, glove_model):
  try:
    valid_metadata = []

    t_path = os.path.join(config_path.DATA_DIR, 'Transcription', f'{id}_transcript.csv')
    v_path = os.path.join(config_path.DATA_DIR, 'Vision Summary', f'{id}_vision_summary.csv')

    if not os.path.exists(t_path) or not os.path.exists(v_path):
      return []
    
    t_df = pd.read_csv(t_path)

    participant_df, group_df = process_transcription(t_df)

    for _, row in group_df.iterrows():
      count = row['count']

      word_list = " ".join(participant_df.loc[participant_df['count']==count].value.tolist()).split()
      valid_words = []
      for word in word_list:
        try:
          word = word.strip()
          if "'" in word:
            word = word.split("'")[0]
          if '<' in word or '>' in word or '_' in word or '[' in word or ']' in word or word in stop_words_list:
            pass
          elif word == "":
            pass
          else:
            valid_words.append(glove_model[word].tolist())
        except KeyError:
          logger.error(f"{word} is not in the model.")
      
      if len(valid_words) > 0:
        valid_metadata.append({
          'participant_id': id,
          'group_count': count,
          'start_time': row['start_time'],
          'stop_time': row['stop_time']
          })
  except Exception as e:
    logger.error(f"Error processing {id}: {e}")
  
  return valid_metadata

def prepare_metadata(ids, stop_words_list, glove_model):
  all_metadata = []
  for id in tqdm(ids, desc="Scanning Dataset Indices"):
    meta = get_valid_indices(id, stop_words_list, glove_model)
    all_metadata.extend(meta)
  return all_metadata

if __name__=="__main__":
  train_transcription_dataset = []
  train_vision_dataset = []
  val_transcription_dataset = []
  val_vision_dataset = []
  
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

  all_metadata = prepare_metadata(train_id+val_id, stop_words_list, glove_model)
  logger.info("Metadata Ready")