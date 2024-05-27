import os
import torch
from language_model import WordEmbedding

# with open(os.path.join('data_RAD', 'embed_tfidf_weights_.pkl'), 'rb') as f:
#     w_emb_ = torch.load(f)
# with open(os.path.join('data_RAD', 'embed_tfidf_weights.pkl'), 'rb') as f:
#     w_emb = torch.load(f)
# # with open(os.path.join('/ssd003/projects/aieng/multimodal/datasets/VQARAD/', 'embed_tfidf_weights.pkl'), 'rb') as f:
# #     w_emb_n = torch.load(f)

# print(w_emb_)
# print(w_emb)
# # print(w_emb_n)

# params_ = list(w_emb_.parameters())
# params = list(w_emb.parameters())
# # params_n = list(w_emb_n.parameters())

# for i, p in enumerate(params_):
#     print(p.data)
#     print(params[i].data)
#     # print(params_n[i].data)
#     print(torch.equal(p.data, params[i].data))
#     # print(torch.equal(p.data, params_n[i].data))


# # generate a state_dict of the previous embed_tfidf_weights.pkl
# with open(os.path.join('data_RAD', 'embed_tfidf_weights.pkl'), 'rb') as f:
#     w_emb = torch.load(f)

# torch.save(w_emb.state_dict(), os.path.join('data_RAD', 'embed_tfidf_weights.pth'))

# w_emb2 = WordEmbedding(1177, 300, .0, "c")
# w_emb2.load_state_dict(torch.load(os.path.join('data_RAD', 'embed_tfidf_weights.pth')))

# params = list(w_emb.parameters())
# params2 = list(w_emb2.parameters())

# for i, p in enumerate(params):
#     print(p.data)
#     print(params2[i].data)
#     # print(params_n[i].data)
#     print(torch.equal(p.data, params2[i].data))
#     # print(torch.equal(p.data, params_n[i].data))
