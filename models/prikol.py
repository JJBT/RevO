import time

import torch
from torch import nn
from models.transformer import SimpleTransformer


class PrikolNet(nn.Module):
    def __init__(self, backbone, pool_shape, embd_dim, n_head, attn_pdrop, resid_pdrop,
                 embd_pdrop, n_layer, out_dim, **kwargs):
        super(PrikolNet, self).__init__()
        self.embd_dim = embd_dim
        self.n_head = n_head
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.n_layer = n_layer
        self.out_dim = out_dim

        self.backbone = backbone

        self.center_pool = CenterPool(img_size=pool_shape, embd_dim=self.embd_dim)
        self.transformer = SimpleTransformer(self.embd_dim, self.n_head, self.attn_pdrop, self.resid_pdrop,
                                             self.embd_pdrop, self.n_layer, self.out_dim)

    def forward(self, sample):
        """
        Args:
            sample (dict): input sample
                sampe['q_img'] (torch.Tensor): batch of query images (BxCxHxW)
                sampe['s_imgs'] (torch.Tensor): batch of support set images (BxKxCxHxW)
                sampe['s_bboxes'] (List[List[List[float, ..]]]): batch of bbox coordinates for support
                                                                 set images (conditionally BxKxVx4, K - length of support set
                                                                                                    V - number of instances per image)
        """
        q_img = sample['q_img']
        s_imgs = sample['s_imgs']
        s_bboxes = sample['s_bboxes']

        B, K, C_s, W_s, H_s = s_imgs.shape
        s_imgs = s_imgs.view((B * K, C_s, W_s, H_s))  # B x K x C x W x H -> B*K x W x H

        # Getting feature maps of query and support images
        # B x C_ x W_ x H_ -> B x C_fm x W_fm x H_fm
        layer = 'output'
        fm = self.backbone(torch.cat([q_img, s_imgs]))[layer]
        q_feature_map = fm[:B]
        s_feature_maps = fm[B:]

        # Getting sequences of query feature vectors images and support object instances feature vectors
        _, C_q_fm, W_q_fm, H_q_fm = q_feature_map.shape
        q_feature_vectors = q_feature_map.permute(0, 2, 3, 1).contiguous().view(-1, W_q_fm * H_q_fm, C_q_fm)  # B x C_fm x H_fm x W_fm -> B x W_fm * H_fm x C_fm
        # q_feature_vectors = torch.cat([q_feature_vectors, torch.zeros(*q_feature_vectors.shape[:-1], 4,
        #                                                               device=q_feature_vectors.device)], dim=2)

        s_feature_vectors_listed = self.center_pool(s_feature_maps, s_bboxes)


        # Shaping batch from feature vector sequences
        # and creating mask for transformer
        seq, mask = self._collate_fn(q_feature_vectors, s_feature_vectors_listed)
        seq = seq.to(fm.device)
        mask = mask.to(fm.device)

        # Getting predictions
        seq_out = self.transformer({'x': seq, 'mask': mask})  # -> B x C_fm x (W_fm * H_fm + N_padded)
        preds = seq_out[:, -(W_q_fm * H_q_fm):]  # only query vectors are responsible for prediction
        preds = preds.squeeze(-1)                # -> B x output_dim x W_fm * H_fm
        preds = preds.view(-1, H_q_fm, W_q_fm, preds.shape[-1])
        return preds

    def _collate_fn(self, q_vectors, s_vectors_listed):
        """
        Collates batch from a sequences of feature vectors with varying length and creates mask for transformer
        Args:
            q_vectors (torch.Tensor): sequence of query feature vectors (H_fm * W_fm x embd_dim)
            s_vectors_listed (List[List[List[float]]]): list of support feature vectors

        """
        batch_size, q_seq_len, embd_dim = q_vectors.shape
        max_s_seq_len = max(map(len, s_vectors_listed))

        batch = torch.zeros(batch_size, max_s_seq_len + q_seq_len, embd_dim)
        mask = torch.zeros(size=(batch_size, max_s_seq_len + q_seq_len, max_s_seq_len + q_seq_len), dtype=torch.bool)
        for i, s_vectors in enumerate(s_vectors_listed):
            s_seq_len = len(s_vectors)
            batch[i, :s_seq_len] = torch.stack(s_vectors)
            mask[i, -q_seq_len:, :s_seq_len] = 1

        batch[:, -q_seq_len:] = q_vectors
        mask[:, -q_seq_len:, -q_seq_len:] = torch.diag(torch.ones(q_seq_len))

        return batch, mask


class PrikolNetClassification(nn.Module):
    def __init__(self, backbone, pool_shape, embd_dim, n_head, attn_pdrop, resid_pdrop,
                 embd_pdrop, n_layer, out_dim, **kwargs):
        super().__init__()
        self.embd_dim = embd_dim
        self.n_head = n_head
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.n_layer = n_layer
        self.out_dim = out_dim

        self.backbone = backbone

        self.transformer = SimpleTransformer(self.embd_dim, self.n_head, self.attn_pdrop, self.resid_pdrop,
                                             self.embd_pdrop, self.n_layer, self.out_dim)

    def forward(self, sample):
        """
        Args:
            sample (dict): input sample
                sampe['q_img'] (torch.Tensor): batch of query images (BxCxHxW)
                sampe['s_imgs'] (torch.Tensor): batch of support set images (BxKxCxHxW)
                sampe['s_bboxes'] (List[List[List[float, ..]]]): batch of bbox coordinates for support
                                                                 set images (conditionally BxKxVx4, K - length of support set
                                                                                                    V - number of instances per image)
        """
        q_img = sample['q_img']
        s_imgs = sample['s_imgs']

        B, K, C_s, W_s, H_s = s_imgs.shape
        s_imgs = s_imgs.view((B * K, C_s, W_s, H_s))  # B x K x C x W x H -> B*K x W x H

        # Getting feature maps of query and support images
        # B x C_ x W_ x H_ -> B x C_fm x W_fm x H_fm
        layer = 'output'
        fm = self.backbone(torch.cat([q_img, s_imgs]))[layer]
        fm = fm.squeeze(-1).squeeze(-1)

        q_feature_vectors = fm[:B]
        s_feature_vectors = fm[B:]

        # Shaping batch from feature vector sequences
        # and creating mask for transformer
        seq, mask = self._collate_fn(q_feature_vectors, s_feature_vectors)
        seq = seq.to(fm.device)
        mask = mask.to(fm.device)

        # Getting predictions
        seq_out = self.transformer({'x': seq, 'mask': mask})  # -> B x C_fm x (W_fm * H_fm + N_padded)
        preds = seq_out[:, -1]  # only query vectors are responsible for prediction
        preds = preds.squeeze(-1)                # -> B x output_dim x W_fm * H_fm
        return preds

    def _collate_fn(self, q_vectors, s_vectors):
        """
        Collates batch from a sequences of feature vectors with varying length and creates mask for transformer
        Args:
            q_vectors (torch.Tensor): sequence of query feature vectors (H_fm * W_fm x embd_dim)
            s_vectors_listed (List[List[List[float]]]): list of support feature vectors

        """
        batch_size, embd_dim = q_vectors.shape
        K = s_vectors.shape[0] // batch_size

        batch = torch.zeros(batch_size, K + 1, embd_dim)
        mask = torch.zeros(size=(batch_size, K + 1, K + 1), dtype=torch.bool)

        batch[:, :K] = torch.stack(torch.split(s_vectors, K))
        batch[:, -1] = q_vectors
        mask[:, -1, :] = torch.ones(K + 1)

        return batch, mask


class CenterPool(nn.Module):
    """
    Pulls out feature vectors of object instances given feature map and bbox coordinates
    """
    def __init__(self, img_size, embd_dim):
        """
        Args:
            img_size: image original size
        """
        super(CenterPool, self).__init__()
        self.img_size = img_size
        self.embd_dim = embd_dim
        self.label_fuser = LabelFuser(in_dim=4, out_dim=self.embd_dim)

    def forward(self, input, bboxes):
        """
        Args:
            input (torch.Tensor): feature map (B x C_fm x W_fm x H_fm)
            bboxes (List[List[List[float]]]): list of bbox coordinates

        Returns:
            (List[List[torch.Tensor]]): list of support feature vectors
        """
        img_w, img_h = self.img_size[-2:]
        fm_w, fm_h = input.shape[-2:]
        cell_w, cell_h = img_w / fm_w, img_h / fm_h

        batch_size = len(bboxes)
        k_shot = len(bboxes[0])
        feature_vectors_listed = []

        for i in range(batch_size):
            feature_vectors = []
            for j in range(k_shot):
                for bbox in bboxes[i][j]:
                    img_xc, img_yc = bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2
                    cell_x, cell_y = int(img_xc / cell_w), int(img_yc / cell_h)
                    feature_vector = input[i * k_shot + j, :, cell_y, cell_x]
                    label = [(img_xc - cell_x * cell_w) / cell_w,
                             (img_yc - cell_y * cell_h) / cell_h,
                             bbox[2] / img_w,
                             bbox[3] / img_h]
                    # label = torch.logit(torch.as_tensor(label, device=input.device), eps=1e-12)
                    # feature_vector = torch.cat([feature_vector, label])

                    feature_vector = self.label_fuser(
                        feature_vector, torch.tensor(label, dtype=feature_vector.dtype, device=feature_vector.device)
                    )
                    feature_vectors.append(feature_vector)

            feature_vectors_listed.append(feature_vectors)

        return feature_vectors_listed


class LabelFuser(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LabelFuser, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, feature_vector, label):
        out = feature_vector + self.linear(label)
        return out
