# import os

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.join(BASE_DIR, '..')

# DATA_DIR = os.path.join(ROOT_DIR, 'data')
# RAW_DATA_DIR = os.path.join(DATA_DIR, 'Raw Data')

# MODEL_DIR = os.path.join(ROOT_DIR, 'model')

# path_config.py 수정
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '..')

# [수정] 프로젝트 내부가 아닌 바탕화면의 data 폴더를 가리키도록 설정
DATA_DIR = "/Users/yuyeoeun/Desktop/data"

RAW_DATA_DIR = os.path.join(DATA_DIR, 'Raw Data')
MODEL_DIR = os.path.join(ROOT_DIR, 'model')