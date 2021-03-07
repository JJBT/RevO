import torch
from torch import nn
from .backbone import resnet_backbone
from .transformer import SimpleTransformer


class PrikolNet(nn.Module):
    def __init__(self, backbone_name, backbone_pratrained,
                 backbone_trainable_layers, backbone_returned_layers,
                 pool_shape, embd_dim, n_head, attn_pdrop, resid_pdrop,
                 embd_pdrop, n_layer, out_dim):
        super(PrikolNet, self).__init__()
        self.backbone_name = backbone_name
        self.backbone_pretrained = backbone_pratrained
        self.backbone_trainable_layers = backbone_trainable_layers
        self.backbone_returned_layers = backbone_returned_layers
        self.embd_dim = embd_dim
        self.n_head = n_head
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.n_layer = n_layer
        self.out_dim = out_dim

        self.backbone = resnet_backbone(self.backbone_name,
                                        self.backbone_pretrained,
                                        self.backbone_trainable_layers,
                                        self.backbone_returned_layers)

        self.center_pool = CenterPool(original_shape=pool_shape)
        self.transformer = SimpleTransformer(self.embd_dim, self.n_head, self.attn_pdrop, self.resid_pdrop,
                                             self.embd_pdrop, self.n_layer, self.out_dim)

    def forward(self, sample):
        q_img = sample['q_img']
        s_imgs = sample['s_imgs']
        s_bboxes = sample['s_bboxes']

        B, K, C_s, W_s, H_s = s_imgs.shape
        s_imgs = s_imgs.view((B * K, C_s, W_s, H_s))

        layer = 'layer' + str(self.backbone_returned_layers)
        q_feature_map = self.backbone(q_img)[layer]
        s_feature_maps = self.backbone(s_imgs)[layer]

        _, C_q_fm, W_q_fm, H_q_fm = q_feature_map.shape
        q_feature_vectors = q_feature_map.permute(0, 2, 3, 1).contiguous().view(-1, W_q_fm * H_q_fm, C_q_fm)
        s_feature_vectors_listed = self.center_pool(s_feature_maps, s_bboxes)

        seq, mask = self._collate_fn(q_feature_vectors, s_feature_vectors_listed)
        seq_out = self.transformer({'x': seq, 'mask': mask})
        preds = seq_out[:, -(W_q_fm * H_q_fm):]

        return preds

    def _collate_fn(self, q_vectors, s_vectors_listed):
        batch_size, q_seq_len, embd_dim = q_vectors.shape
        max_s_seq_len = max(map(len, s_vectors_listed))

        batch = torch.zeros(batch_size, max_s_seq_len + q_seq_len, embd_dim)
        mask = torch.zeros(size=(batch_size, max_s_seq_len + q_seq_len, max_s_seq_len + q_seq_len), dtype=torch.bool)
        for i, s_vectors in enumerate(s_vectors_listed):
            s_seq_len = len(s_vectors)
            batch[i, :s_seq_len] = torch.stack(s_vectors)
            mask[i, -q_seq_len:, :s_seq_len] = 1

        batch[:, -q_seq_len:] = q_vectors

        return batch, mask


class CenterPool(nn.Module):
    def __init__(self, original_shape):
        super(CenterPool, self).__init__()
        self.original_shape = original_shape

    def forward(self, input, bboxes):
        orig_w, orig_h = self.original_shape[-2:]
        fm_w, fm_h = input.shape[-2:]
        cell_w, cell_h = orig_w / fm_w, orig_h / fm_h

        batch_size = len(bboxes)
        k_shot = len(bboxes[0])
        feature_vectors_listed = []

        for i in range(batch_size):
            feature_vectors = []
            for j in range(k_shot):
                for bbox in bboxes[i][j]:
                    img_x, img_y = bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2
                    cell_x, cell_y = int(img_x / cell_w), int(img_y / cell_h)
                    feature_vector = input[i * k_shot + j, :, cell_y, cell_x]
                    feature_vectors.append(feature_vector)

            feature_vectors_listed.append(feature_vectors)

        return feature_vectors_listed
