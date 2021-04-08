import torch
from torch import nn
from models.backbone import resnet_backbone
from models.transformer import SimpleTransformer


class PrikolNet(nn.Module):
    def __init__(self, backbone, pool_shape, embd_dim, n_head, attn_pdrop, resid_pdrop,
                 embd_pdrop, n_layer, out_dim, device, **kwargs):
        super(PrikolNet, self).__init__()
        self.embd_dim = embd_dim
        self.n_head = n_head
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.n_layer = n_layer
        self.out_dim = out_dim
        self.device = device

        self.backbone = backbone

        self.center_pool = CenterPool(original_size=pool_shape)
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

        q_img = q_img.to(self.device)
        s_imgs = s_imgs.to(self.device)

        B, K, C_s, W_s, H_s = s_imgs.shape
        s_imgs = s_imgs.view((B * K, C_s, W_s, H_s))  # B x K x C x W x H -> B*K x W x H

        # Getting feature maps of query and support images
        # B x C_ x W_ x H_ -> B x C_fm x W_fm x H_fm
        layer = 'output'
        q_feature_map = self.backbone(q_img)[layer]
        s_feature_maps = self.backbone(s_imgs)[layer]

        # Getting sequences of query feature vectors images and support object instances feature vectors
        _, C_q_fm, W_q_fm, H_q_fm = q_feature_map.shape
        q_feature_vectors = q_feature_map.permute(0, 2, 3, 1).contiguous().view(-1, W_q_fm * H_q_fm, C_q_fm)  # B x C_fm x H_fm x W_fm -> B x W_fm * H_fm x C_fm
        s_feature_vectors_listed = self.center_pool(s_feature_maps, s_bboxes)

        # Shaping batch from feature vector sequences
        # and creating mask for transformer
        seq, mask = self._collate_fn(q_feature_vectors, s_feature_vectors_listed)
        seq = seq.to(self.device)
        mask = mask.to(self.device)

        # Getting predictions
        seq_out = self.transformer({'x': seq, 'mask': mask})  # -> B x C_fm x (W_fm * H_fm + N_padded)
        preds = seq_out[:, -(W_q_fm * H_q_fm):]  # only query vectors are responsible for prediction
        preds = preds.squeeze(-1)                # -> B x output_dim x W_fm * H_fm

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


class CenterPool(nn.Module):
    """
    Pulls out feature vectors of object instances given feature map and bbox coordinates
    """
    def __init__(self, original_size):
        """
        Args:
            original_size: image original size
        """
        super(CenterPool, self).__init__()
        self.original_size = original_size

    def forward(self, input, bboxes):
        """
        Args:
            input (torch.Tensor): feature map (B x C_fm x W_fm x H_fm)
            bboxes (List[List[List[float]]]): list of bbox coordinates

        Returns:
            (List[List[torch.Tensor]]): list of support feature vectors
        """
        orig_w, orig_h = self.original_size[-2:]
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
