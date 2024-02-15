# import _init_paths
# from language.language_model import WordEmbedding, QuestionEmbedding
# from classifier import SimpleClassifier
# from network.connect import FCNet
# from network.connect import BCNet
# from network.counting import Counter
# from utils.utils import tfidf_loading
# from network.maml import SimpleCNN
# from network.auto_encoder import Auto_Encoder_Model
# from torch.nn.utils.weight_norm import weight_norm
# from language.classify_question import typeAttention 
# import os


import torch
import clip
import os
# import pickle
import _pickle as cPickle
# from PIL import Image

# load clip
CLIP_VISION_ENCODER = 'RN50'
device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess = clip.load(CLIP_VISION_ENCODER, jit=False, device=device)
# print(model_clip)

# load PubMedCLIP pretrained weights
pretrained_path = "saved_models/PubMedCLIP/PubMedCLIP_RN50.pth"
checkpoint = torch.load(pretrained_path)
model_clip.load_state_dict(checkpoint['state_dict'])
model_clip = model_clip.float()
# print(model_clip)

# load images of size 84x84
# data_v_path = 'data/data_rad'
# with open(os.path.join(data_v_path, 'images84x84.pkl'), 'rb') as file:
#     images84 = pickle.load(file)

dataroot = 'data_RAD'
if CLIP_VISION_ENCODER == "RN50x4":
    images_path = os.path.join(dataroot, 'images288x288.pkl')
else:
    images_path = os.path.join(dataroot, 'images250x250.pkl')
clip_images_data = cPickle.load(open(images_path, 'rb'))

clip_images_data = torch.from_numpy(clip_images_data)
clip_images_data = clip_images_data.type('torch.FloatTensor')

# entries = _load_dataset(dataroot, name, img_id2idx, label2ans)

# if self.cfg.TRAIN.VISION.CLIP_VISION_ENCODER == "RN50x4":
#     clip_images_data = self.clip_images_data[entry['image']].reshape(3*288*288)
# else:
#     clip_images_data = self.clip_images_data[entry['image']].reshape(3*250*250)

img_batch = clip_images_data[:32]

if CLIP_VISION_ENCODER == "RN50x4":
    img_batch = img_batch.reshape(img_batch.shape[0], 3, 288, 288)
else:
    img_batch = img_batch.reshape(img_batch.shape[0], 3, 250, 250)
img_batch = img_batch.to(device)

clip_v_emb = model_clip.encode_image(img_batch).unsqueeze(1)
print(clip_v_emb.size())

# if not cfg.TRAIN.VISION.CLIP_ORG:
# checkpoint = torch.load(cfg.TRAIN.VISION.CLIP_PATH)
# self.clip.load_state_dict(checkpoint['state_dict'])
# self.clip = self.clip.float()