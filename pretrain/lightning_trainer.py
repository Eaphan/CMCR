import random
import os
import re
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import MinkowskiEngine as ME
import pytorch_lightning as pl
from pretrain.criterion import NCELoss
from pytorch_lightning.utilities import rank_zero_only
from torch_geometric.nn import radius as search_radius, knn as search_knn

from functools import partial
import torch.distributed as dist

def orthogonalization_loss(feature1, feature2):
    """
    计算两个特征矩阵的正交化损失。
    
    参数:
    - feature1: Tensor, shape (N, C) 模态共享特征
    - feature2: Tensor, shape (N, C) 模态特定特征
    
    返回:
    - loss: Tensor, 正交化损失值
    """
    # 特征矩阵的转置
    feature1_T = feature1.transpose(0, 1)  # shape (C, N)
    feature2_T = feature2.transpose(0, 1)  # shape (C, N)
    
    # 计算特征之间的内积矩阵
    inner_product_matrix = torch.matmul(feature1_T, feature2)  # shape (C, C)
    
    # 计算 Frobenius 范数 (L2 范数)
    loss = torch.norm(inner_product_matrix, p='fro')
    
    return loss


class LightningPretrain(pl.LightningModule):
    def __init__(self, model_points, model_images, config):
        super().__init__()
        self.model_points = model_points
        self.model_images = model_images
        self.image_recons_decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),  # [24, 32, 224, 416]
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),  # [24, 16, 224, 416]
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),   # [24, 3, 224, 416]
            nn.Sigmoid()  # Assuming input is normalized between [0, 1]
        )
        self._config = config
        self.losses = config["losses"]
        self.train_losses = []
        self.val_losses = []
        self.num_matches = config["num_matches"]
        self.batch_size = config["batch_size"]
        self.num_epochs = config["num_epochs"]
        self.superpixel_size = config["superpixel_size"]
        self.epoch = 0
        if config["resume_path"] is not None:
            self.epoch = int(
                re.search(r"(?<=epoch=)[0-9]+", config["resume_path"])[0]
            )
        self.criterion = NCELoss(temperature=config["NCE_temperature"])
        self.working_dir = os.path.join(config["working_dir"], config["datetime"])
        if os.environ.get("LOCAL_RANK", 0) == 0:
            os.makedirs(self.working_dir, exist_ok=True)

        # ad hoc, choose radius search
        self.radius = 1.0
        self.search_function = partial(search_radius, r=self.radius)
        latent_size = 128
        self.occ_fc_in = torch.nn.Linear(64+3, latent_size)
        # mlp_layers = [torch.nn.Linear(latent_size, latent_size) for _ in range(2)]
        mlp_layers = []
        for _ in range(2):
            mlp_layers.append(torch.nn.ReLU())
            mlp_layers.append(torch.nn.Linear(latent_size, latent_size))
        self.occ_mlp_layers = nn.ModuleList(mlp_layers)
        self.occ_fc_out = torch.nn.Linear(latent_size, 2)

        # FOR CODEBOOK
        # init_bound = 1 / 400
        n_embeddings = 400
        embedding_dim = config['model_n_out']
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-1, 1) # todo
        embedding = F.normalize(embedding, p=2, dim=1)
        # if dist.get_rank() == 0:
        #     embedding = torch.Tensor(n_embeddings, embedding_dim)
        #     embedding.uniform_(-1, 1)
        #     embedding = F.normalize(embedding, p=2, dim=1)
        # else:
        #     embedding = torch.empty(n_embeddings, embedding_dim)
        # dist.broadcast(embedding, src=0) # from rank 0 to other devices

        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())
        self.register_buffer("unactivated_count", -torch.ones(n_embeddings))# unactivated:-1

        self.decay = 0.999
        self.commitment_cost = 0.25
        self.epsilon = 1e-5

    # def on_fit_start(self):
    #     # Only on rank 0 do we initialize the embedding
    #     print(f'#### before broadcat, rank={self.global_rank}, emb.mean={self.embedding.mean()}, emb.std={self.embedding.std()}')
    #     # if self.global_rank == 0:
    #     #     self.embedding.uniform_(-1, 1)
    #     #     self.embedding = F.normalize(self.embedding, p=2, dim=1)

    #     # Broadcasting the initialized buffer from rank 0 to all devices
    #     dist.broadcast(self.embedding, src=0)
    #     print(f'#### rank={self.global_rank}, emb.mean={self.embedding.mean()}, emb.std={self.embedding.std()}')

    def to(self, device):
        super(LightningPretrain, self).to(device)
        
        # 手动将 buffer 移动到指定设备
        self.embedding = self.embedding.to(device)
        self.ema_count = self.ema_count.to(device)
        self.ema_weight = self.ema_weight.to(device)
        self.unactivated_count = self.unactivated_count.to(device)
        print("### device=", device)
        return self

    def configure_optimizers(self):
        # optimizer = optim.SGD(
        #     self.parameters(),
        #     lr=self._config["lr"],
        #     momentum=self._config["sgd_momentum"],
        #     dampening=self._config["sgd_dampening"],
        #     weight_decay=self._config["weight_decay"],
        # )
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.num_epochs)
        # return [optimizer], [scheduler]
        optimizer = optim.AdamW(self.parameters(), lr=self._config["lr"],)
        return [optimizer]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def training_step(self, batch, batch_idx):
        sparse_input = ME.SparseTensor(batch["sinput_F"], batch["sinput_C"])
        # output_points = self.model_points(sparse_input).F
        r = self.model_points(sparse_input)
        output_points, output_points_occ = r[0].F, r[1].F
        # output_points_occ_norm = torch.norm(output_points_occ, p=2, dim=1, keepdim=True)
        output_points_occ_norm = F.normalize(output_points_occ, p=2, dim=1)
        loss_orth = orthogonalization_loss(output_points, output_points_occ_norm) / (64 ** 2)
        self.log("loss_orth", loss_orth.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.model_images.eval()
        self.model_images.decoder.train()
        output_images, output_images_recon = self.model_images(batch["input_I"])

        output_points_deatch = output_points.detach()
        p_distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                            torch.sum(output_points_deatch ** 2, dim=1, keepdim=True),
                            output_points_deatch, self.embedding.t(),
                            alpha=-2.0, beta=1.0)
        p_indices = torch.argmin(p_distances.double(), dim=-1)  # [BxT,1]
        p_quantized = F.embedding(p_indices, self.embedding)
        p_quantized = output_points + (p_quantized - output_points).detach()
        output_points_occ = output_points_occ + p_quantized

        output_images_flatten = output_images.permute(0, 2, 3, 1).contiguous().view(-1, 64)
        output_images_deatch = output_images_flatten.detach()
        v_distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                            torch.sum(output_images_deatch ** 2, dim=1, keepdim=True),
                            output_images_deatch, self.embedding.t(),
                            alpha=-2.0, beta=1.0)
        v_indices = torch.argmin(v_distances.double(), dim=-1)  # [BxT,1]
        v_quantized = F.embedding(v_indices, self.embedding)
        v_quantized = output_images_flatten + (v_quantized - output_images_flatten).detach()
        bs_cam, c, image_feature_h, image_feature_w = output_images.shape
        v_quantized = v_quantized.view(bs_cam, image_feature_h, image_feature_w, c).permute(0, 3, 1, 2).contiguous()
        output_images_recon = output_images_recon + v_quantized
        output_images_recon = self.image_recons_decoder(output_images_recon)
        images_recon_loss = F.mse_loss(output_images_recon, batch["target_I"])

        # occupancy estimation start
        pos_target = batch["pos_non_manifold"][:, 1:]
        batch_target = batch["pos_non_manifold"][:, 0].int()
        batch_source = torch.cat([torch.zeros(x.shape[0], device=x.device)+i for i,x in enumerate(batch['pc'])], 0)
        pos_source = torch.cat(batch['pc'], 0)
        b_voxel_nums = torch.bincount(batch["sinput_C"][:, 0])
        voxel_offsets = torch.cumsum(b_voxel_nums, dim=0) - b_voxel_nums
        inverse_indexes_global = torch.cat([
            batch['inverse_indexes'][i] + voxel_offsets[i]
            for i in range(len(batch['pc']))
        ])
        row, col = self.search_function(x=pos_source, y=pos_target, batch_x=batch_source.long(), batch_y=batch_target.long())
        pos_relative = pos_target[row] - pos_source[col]
        latents_relative = output_points_occ[inverse_indexes_global][col]
        query_points_x = torch.cat([latents_relative, pos_relative], dim=1)
        query_points_x = self.occ_fc_in(query_points_x.contiguous())
        for i, l in enumerate(self.occ_mlp_layers):
            query_points_x = l(query_points_x)
        query_points_out = self.occ_fc_out(query_points_x)
        recons_loss = F.binary_cross_entropy_with_logits(query_points_out[:, 0], batch["occupancies"][row].float())
        query_points_intensity = batch["intensities_non_manifold"][row, 0]
        intensity_mask = (query_points_intensity >= 0)
        intensity_loss = F.l1_loss(query_points_out[:, 1][intensity_mask], query_points_intensity[intensity_mask])
        # occupancy estimation done

        # each loss is applied independtly on each GPU
        cl_losses = [
            getattr(self, loss)(batch, output_points, output_images)
            for loss in self.losses
        ]
        cl_loss = torch.mean(torch.stack(cl_losses))
        point_embedding_loss, image_bedding_loss, cmc_loss, _, _ = self.loss_codebook(batch, output_points, output_images)
        
        self.log("p_embedding_loss", point_embedding_loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("v_embedding_loss", image_bedding_loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("cmc_loss", cmc_loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True)

        self.log("cl_loss", cl_loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        # cl_loss = 0
        self.log("images_recon_loss", images_recon_loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("recons_loss", recons_loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("intensity_loss", intensity_loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("lr", self.optimizers().state_dict()['param_groups'][0]['lr'], on_step=True, on_epoch=False, prog_bar=True, logger=True)
        loss = cl_loss + images_recon_loss + recons_loss + intensity_loss + loss_orth * 0.1 + point_embedding_loss + image_bedding_loss + cmc_loss

        torch.cuda.empty_cache()
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size
        )
        self.train_losses.append(loss.detach().cpu())
        return loss

    def loss(self, batch, output_points, output_images):
        pairing_points = batch["pairing_points"]
        pairing_images = batch["pairing_images"]
        idx = np.random.choice(pairing_points.shape[0], self.num_matches, replace=False)
        k = output_points[pairing_points[idx]]
        m = tuple(pairing_images[idx].T.long())
        q = output_images.permute(0, 2, 3, 1)[m]
        return self.criterion(k, q)

    def loss_codebook(self, batch, output_points, output_images):
        pairing_points = batch["pairing_points"]
        pairing_images = batch["pairing_images"]
        # paired_point_feats = output_points[pairing_points]
        # m = tuple(pairing_images.T.long())
        # paired_image_feats = output_images.permute(0, 2, 3, 1)[m]

        # still sampling, for cmcm loss
        idx = np.random.choice(pairing_points.shape[0], self.num_matches, replace=False)
        paired_point_feats = output_points[pairing_points[idx]]
        m = tuple(pairing_images[idx].T.long())
        paired_image_feats = output_images.permute(0, 2, 3, 1)[m]

        paired_point_feats_detach = paired_point_feats.detach()
        paired_image_feats_detach = paired_image_feats.detach()

        p_distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                            torch.sum(paired_point_feats_detach ** 2, dim=1, keepdim=True),
                            paired_point_feats_detach, self.embedding.t(),
                            alpha=-2.0, beta=1.0)

        v_distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                  torch.sum(paired_image_feats_detach ** 2, dim=1, keepdim=True),
                                  paired_image_feats_detach, self.embedding.t(),
                                  alpha=-2.0, beta=1.0)  # [BxT, M]

        p_distances_gradient = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                  torch.sum(paired_point_feats ** 2, dim=1, keepdim=True),
                                  paired_point_feats, self.embedding.t(),
                                  alpha=-2.0, beta=1.0)  # [BxT, M]

        v_distances_gradient = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                           torch.sum(paired_image_feats ** 2, dim=1, keepdim=True),
                                           paired_image_feats, self.embedding.t(),
                                           alpha=-2.0, beta=1.0)  # [BxT, M]

        p_ph = F.softmax(-torch.sqrt(p_distances_gradient), dim=1)
        v_ph = F.softmax(-torch.sqrt(v_distances_gradient), dim=1)
        p_pH = p_ph
        v_pH = v_ph

        # todo
        Scode_pv = p_pH @ torch.log(v_pH.t() + 1e-10) + v_pH @ torch.log(p_pH.t() + 1e-10)
        MaxScode_pv = torch.max(-Scode_pv)
        EScode_pv = torch.exp(Scode_pv + MaxScode_pv)
        
        EScode_sumdim1_pv = torch.sum(EScode_pv, dim=1)
        Lcmcm_pv = 0
        for i in range(self.num_matches):
            Lcmcm_pv -= torch.log(EScode_pv[i, i] / (EScode_sumdim1_pv[i] + self.epsilon))
        Lcmcm_pv /= self.num_matches

        M, _ = self.embedding.size()
        p_indices = torch.argmin(p_distances.double(), dim=-1)  # [BxT,1]
        p_encodings = F.one_hot(p_indices, M).float()  # [BxT, M]
        p_quantized = F.embedding(p_indices, self.embedding)
        p_quantized = p_quantized.view_as(paired_point_feats)  # [BxT,D]->[B,T,D]

        v_indices = torch.argmin(v_distances.double(), dim=-1)  # [BxT,1]
        v_encodings = F.one_hot(v_indices, M).float()  # [BxT, M]
        v_quantized = F.embedding(v_indices, self.embedding)  
        v_quantized = v_quantized.view_as(paired_image_feats)  # [BxT,D]->[B,T,D]
        
        # p_indices_reshape = p_indices.reshape(B, T)
        # v_indices_reshape = v_indices.reshape(B, T)
        # p_indices_mode = torch.mode(p_indices_reshape, dim=-1, keepdim=False)
        # v_indices_mode = torch.mode(v_indices_reshape, dim=-1, keepdim=False)

        # equal_item = (p_indices_mode.values == v_indices_mode.values)
        # equal_num = equal_item.sum()
        equal_num = (p_indices == v_indices).sum()
        
        # point cloud
        self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(p_encodings, dim=0)
        n = torch.sum(self.ema_count)
        self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n
        p_dw = torch.matmul(p_encodings.t(), paired_point_feats_detach)
        pv_dw = torch.matmul(p_encodings.t(), paired_image_feats_detach)
        self.ema_weight = self.decay * self.ema_weight + 0.5*(1 - self.decay) * p_dw + 0.5*(1 - self.decay) * pv_dw

        # visual
        self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(v_encodings, dim=0)
        n = torch.sum(self.ema_count)
        self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n
        v_dw = torch.matmul(v_encodings.t(), paired_image_feats_detach)
        vp_dw = torch.matmul(v_encodings.t(), paired_point_feats_detach)
        self.ema_weight = self.decay * self.ema_weight + 0.5*(1 - self.decay) * v_dw + 0.5*(1 - self.decay) * vp_dw

        dist.all_reduce(self.ema_weight, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.ema_count, op=dist.ReduceOp.SUM)
        self.ema_weight /= dist.get_world_size()
        self.ema_count /= dist.get_world_size()
        self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)
        self.embedding = F.normalize(self.embedding, p=2, dim=1)

        '''
        self.unactivated_count = self.unactivated_count + 1
        for indice in p_indices:
            self.unactivated_count[indice.item()] = 0
        for indice in v_indices:
            self.unactivated_count[indice.item()] = 0
        activated_indices = []
        unactivated_indices = []
        for i, x in enumerate(self.unactivated_count):
            if x > 300:
                unactivated_indices.append(i)
                self.unactivated_count[i] = 0
            elif x >= 0 and x < 100:
                activated_indices.append(i)
        activated_quantized = F.embedding(torch.tensor(activated_indices,dtype=torch.int32).cuda(), self.embedding)
        for i in unactivated_indices:
            self.embedding[i] = activated_quantized[random.randint(0,len(activated_indices)-1)] + torch.Tensor(64).uniform_(-1/1024, 1/1024).cuda()
        '''

        # cmcm_loss = 0.5 * Lcmcm_pv
        cmcm_loss = Lcmcm_pv

        p_e_latent_loss = F.mse_loss(paired_point_feats, p_quantized.detach())
        pv_e_latent_loss = F.mse_loss(paired_point_feats, v_quantized.detach())
        #p_loss = self.commitment_cost * 1.0 * p_e_latent_loss
        p_loss = self.commitment_cost * 2.0 * p_e_latent_loss + self.commitment_cost * pv_e_latent_loss
        
        v_e_latent_loss = F.mse_loss(paired_image_feats, v_quantized.detach())
        vp_e_latent_loss = F.mse_loss(paired_image_feats, p_quantized.detach())
        #v_loss = self.commitment_cost * 1.0 * v_e_latent_loss
        v_loss = self.commitment_cost * 2.0 * v_e_latent_loss + self.commitment_cost * vp_e_latent_loss
        
        p_quantized = paired_point_feats + (p_quantized - paired_point_feats).detach()
        v_quantized = paired_image_feats + (v_quantized - paired_image_feats).detach()

        p_avg_probs = torch.mean(p_encodings, dim=0)
        p_perplexity = torch.exp(-torch.sum(p_avg_probs * torch.log(p_avg_probs + 1e-10)))
        v_avg_probs = torch.mean(v_encodings, dim=0)
        v_perplexity = torch.exp(-torch.sum(v_avg_probs * torch.log(v_avg_probs + 1e-10)))
        
        return p_loss, v_loss, cmcm_loss, p_perplexity, v_perplexity


    def loss_superpixels_average(self, batch, output_points, output_images):
        # compute a superpoints to superpixels loss using superpixels
        torch.cuda.empty_cache()  # This method is extremely memory intensive
        superpixels = batch["superpixels"]
        pairing_images = batch["pairing_images"]
        pairing_points = batch["pairing_points"]

        superpixels = (
            torch.arange(
                0,
                output_images.shape[0] * self.superpixel_size,
                self.superpixel_size,
                device=self.device,
            )[:, None, None] + superpixels
        )
        m = tuple(pairing_images.cpu().T.long())

        superpixels_I = superpixels.flatten()
        idx_P = torch.arange(pairing_points.shape[0], device=superpixels.device)
        total_pixels = superpixels_I.shape[0]
        idx_I = torch.arange(total_pixels, device=superpixels.device)

        with torch.no_grad():
            one_hot_P = torch.sparse_coo_tensor(
                torch.stack((
                    superpixels[m], idx_P
                ), dim=0),
                torch.ones(pairing_points.shape[0], device=superpixels.device),
                (superpixels.shape[0] * self.superpixel_size, pairing_points.shape[0])
            )

            one_hot_I = torch.sparse_coo_tensor(
                torch.stack((
                    superpixels_I, idx_I
                ), dim=0),
                torch.ones(total_pixels, device=superpixels.device),
                (superpixels.shape[0] * self.superpixel_size, total_pixels)
            )

        k = one_hot_P @ output_points[pairing_points]
        k = k / (torch.sparse.sum(one_hot_P, 1).to_dense()[:, None] + 1e-6)
        q = one_hot_I @ output_images.permute(0, 2, 3, 1).flatten(0, 2)
        q = q / (torch.sparse.sum(one_hot_I, 1).to_dense()[:, None] + 1e-6)

        mask = torch.where(k[:, 0] != 0)
        k = k[mask]
        q = q[mask]

        return self.criterion(k, q)

    def training_epoch_end(self, outputs):
        # if self.epoch > 1:
        #     for param in self.model_images.encoder.parameters():
        #         param.requires_grad = True
        # else:
        #     for param in self.model_images.encoder.parameters():
        #         param.requires_grad = False

        self.epoch += 1
        if self.epoch == self.num_epochs:
            self.save()
        return super().training_epoch_end(outputs)

    def validation_step(self, batch, batch_idx):
        return None
        # sparse_input = ME.SparseTensor(batch["sinput_F"], batch["sinput_C"]) 
        # # output_points = self.model_points(sparse_input).F
        # r = self.model_points(sparse_input)
        # output_points, output_points_occ = r[0].F, r[1].F
        # self.model_images.eval()
        # output_images = self.model_images(batch["input_I"])

        # losses = [
        #     getattr(self, loss)(batch, output_points, output_images)
        #     for loss in self.losses
        # ]
        # loss = torch.mean(torch.stack(losses))
        # self.val_losses.append(loss.detach().cpu())

        # self.log(
        #     "val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size
        # )
        # return loss

    @rank_zero_only
    def save(self):
        path = os.path.join(self.working_dir, "model.pt")
        torch.save(
            {
                "state_dict": self.state_dict(),
                # "model_points": self.model_points.state_dict(),
                # "model_images": self.model_images.state_dict(),
                "epoch": self.epoch,
                "config": self._config,
            },
            path,
        )
