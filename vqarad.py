from __future__ import print_function
import os
import json
import _pickle as cPickle
import numpy as np
import utils
import torch
from torch.utils.data.dataset import Dataset
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)

import clip
from dataset.constants import Modality
from dataset.example import Example

COUNTING_ONLY = False
# Following Trott et al. (ICLR 2018)
#   Interpretable Counting for Visual Question Answering
def is_howmany(q, a, label2ans):
    if 'how many' in q.lower() or \
       ('number of' in q.lower() and 'number of the' not in q.lower()) or \
       'amount of' in q.lower() or \
       'count of' in q.lower():
        if a is None or answer_filter(a, label2ans):
            return True
        else:
            return False
    else:
        return False

def answer_filter(answers, label2ans, max_num=10):
    for ans in answers['labels']:
        if label2ans[ans].isdigit() and max_num >= int(label2ans[ans]):
            return True
    return False

def tokenize_clip(sentence, context_length=77):
    sentence = sentence.lower()
    if "? -yes/no" in sentence:
        sentence = sentence.replace("? -yes/no", "")
    if "? -open" in sentence:
        sentence = sentence.replace("? -open", "")
    if "? - open" in sentence:
        sentence = sentence.replace("? - open", "")
    sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s').replace('...', '').replace('x ray', 'x-ray').replace('.', '')
    tokens = clip.tokenize([sentence], context_length).squeeze()
    return tokens


def _create_entry(img, data, answer):
    if None!=answer:
        answer.pop('image_name')
        answer.pop('qid')
    entry = {
        'qid' : data['qid'],
        'image_name'    : data['image_name'],
        'image'       : img,
        'question'    : data['question'],
        'answer'      : answer,
        'answer_type' : data['answer_type'],
        'question_type': data['question_type'],
        'phrase_type' : data['phrase_type'] if 'phrase_type' in data.keys() else -1
    }
    return entry

def _load_dataset(dataroot, name, img_id2val, label2ans):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test'
    """
    data_path = os.path.join(dataroot, name + 'set.json')
    samples = json.load(open(data_path))
    samples = sorted(samples, key=lambda x: x['qid'])

    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    answers = cPickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['qid'])

    utils.assert_eq(len(samples), len(answers))
    entries = []
    for sample, answer in zip(samples, answers):
        utils.assert_eq(sample['qid'], answer['qid'])
        utils.assert_eq(sample['image_name'], answer['image_name'])
        img_id = sample['image_name']
        if not COUNTING_ONLY or is_howmany(sample['question'], answer, label2ans):
            entries.append(_create_entry(img_id2val[img_id], sample, answer))

    return entries

class VQARad(Dataset):
    def __init__(self, name, args, tokenize_fn):
        super(VQARad, self).__init__()
        self.args = args
        assert name in ['train', 'val', 'test']
        dataroot = args.dataroot
        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        # load imgid2idx
        with open(os.path.join(dataroot, 'imgid2idx.json')) as file:
            self.imgid2idx = json.load(file)

        self.entries = _load_dataset(dataroot, name, self.imgid2idx, self.label2ans)

        # load image data for the encoder module
        images_path = os.path.join(dataroot, args.images_filename)
        print('loading encoder image data from file: '+ images_path)
        with open(images_path, 'rb') as file:
            self.enc_images_data = cPickle.load(file)
        
        # load image data for Auto-encoder module     
        if args.autoencoder:
            # load images
            images_path = os.path.join(dataroot, args.ae_images_filename)
            print('loading DAE image data from file: '+ images_path)
            with open(images_path, 'rb') as file:
                self.ae_images_data = cPickle.load(file)
        
        # tokenization
        self.tokenize(tokenize_fn)
        self.tensorize()
        
        if args.autoencoder:
            self.v_dim = args.image_feat_dim + args.ae_feat_dim
        else:
            self.v_dim = args.image_feat_dim
    
    def tokenize(self, tokenize_fn):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        """
        for entry in self.entries:
            tokens = tokenize_fn(entry['question'])
            entry['q_token'] = tokens

    def tensorize(self):
        # tensorize encoder images
        self.enc_images_data = torch.stack(self.enc_images_data)
        self.enc_images_data = self.enc_images_data.type('torch.FloatTensor')

        # tensorize auto-encoder images
        if self.args.autoencoder:
            self.ae_images_data = torch.stack(self.ae_images_data)
            self.ae_images_data = self.ae_images_data.type('torch.FloatTensor')
        
        # tensorize question tokens, answer labels and scores
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            if None!=answer:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        question = entry['q_token']
        answer = entry['answer']
        answer_type = entry['answer_type']
        question_type = entry['question_type']

        image_data = [0, 0]
        # prepare encoder image
        enc_images_data = self.enc_images_data[entry['image']].reshape(3 * self.args.img_size * self.args.img_size)
        image_data[0] = enc_images_data
        if self.args.autoencoder:
            ae_images_data = self.ae_images_data[entry['image']].reshape(1 * self.args.ae_img_size * self.args.ae_img_size)
            image_data[1] = ae_images_data

        if None!=answer:
            labels = answer['labels']
            scores = answer['scores']
            target = torch.zeros(self.num_ans_candidates)
            if labels is not None:
                target.scatter_(0, labels, scores)
            return Example(
                {
                    Modality.RGB_IMAGE: image_data,
                    Modality.TEXT: question,
                    Modality.RGB_IMAGE_CLASS_LABEL: target,
                    "example_index": index,
                    "qid": entry['qid'],
                    "raw_question": entry['question'],
                    "answer_type": answer_type,
                    "question_type": question_type,
                    "phrase_type": entry['phrase_type']
                }
            )
        else:
            return Example(
                    {
                        Modality.RGB_IMAGE: image_data,
                        Modality.TEXT: question,
                        "example_index": index,
                        "qid": entry['qid'],
                        "raw_question": entry['question'],
                        "answer_type": answer_type,
                        "question_type": question_type,
                        "phrase_type": entry['phrase_type']
                    }
                )

    def __len__(self):
        return len(self.entries)

if __name__=='__main__':
    print('TBI')
