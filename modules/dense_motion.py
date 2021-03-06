from torch import nn
import torch.nn.functional as F
import torch
from modules.util import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid, kp2gaussian, LEFT_EYE_IDX, RIGHT_EYE_IDX, LIP_IDX, ROI_IDX, CONTOUR_IDX

class MeshDenseMotionNetwork(nn.Module):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_kp, num_channels, estimate_occlusion_map=False,
                 scale_factor=1):
        super(MeshDenseMotionNetwork, self).__init__()
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=1,
                                   max_features=max_features, num_blocks=num_blocks)

        self.segments = [LEFT_EYE_IDX, RIGHT_EYE_IDX, LIP_IDX]
        self.num_priors = [5, 5, 10]
        self.mask = nn.Conv2d(self.hourglass.out_filters, sum(self.num_priors) + 1 + len(self.segments), kernel_size=(7, 7), padding=(3, 3))
        if estimate_occlusion_map:
            self.occlusion = nn.Conv2d(self.hourglass.out_filters, 1, kernel_size=(7, 7), padding=(3, 3))
        else:
            self.occlusion = None

        self.num_kp = num_kp
        self.scale_factor = scale_factor

        self.motion_prior = nn.ModuleList()
        
        for i in range(len(self.segments)):
            self.motion_prior.append(nn.Linear(len(self.segments[i]) * 2, self.num_priors[i] * 2))

        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

        self.T = 0.1
    def create_sparse_motions(self, source_image, kp_driving, kp_source):
        """
        Eq 4. in the paper T_{s<-d}(z)
        """
        # driving_z: B x H x W x 1
        driving_z = kp_driving['z']
        source_normed_z = kp_source['normed_z']
        bs, _, h, w = source_image.shape
        identity_grid = make_coordinate_grid((h, w), type=kp_source['value'].type())
        identity_grid = identity_grid.view(1, 1, h, w, 2).repeat(bs, 1, 1, 1, 1)
        identity_3d_grid = torch.cat([identity_grid, driving_z.unsqueeze(1)], dim=4) # 1 x 1 x H x W x 3
        normalized_grid = self.normalize_point(kp_driving['R'], kp_driving['t'], kp_driving['c'], identity_3d_grid)[:, :, :, :, :2] # B x 1 x H x W x 2

        coordinate_grid = normalized_grid - kp_driving['value'].view(bs, self.num_kp, 1, 1, 2)

        driving_to_source = coordinate_grid
        driving_to_source = torch.cat([normalized_grid, driving_to_source], dim=1)
        driving_to_source = torch.cat([driving_to_source, source_normed_z.unsqueeze(1).repeat(1, self.num_kp + 1, 1, 1, 1)], dim=4)
        driving_to_source = self.denormalize_point(kp_source['R'], kp_source['t'], kp_source['c'], driving_to_source)[:, :, :, :, :2]

        sparse_motions = driving_to_source
        return sparse_motions

    def create_deformed_source_image(self, source_image, sparse_motions):
        """
        Eq 7. in the paper \hat{T}_{s<-d}(z)
        """
        bs, _, h, w = source_image.shape
        source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp + 1, 1, 1, 1, 1)
        source_repeat = source_repeat.view(bs * (self.num_kp + 1), -1, h, w)
        sparse_motions = sparse_motions.view((bs * (self.num_kp + 1), h, w, -1))
        sparse_deformed = F.grid_sample(source_repeat, sparse_motions)
        sparse_deformed = sparse_deformed.view((bs, self.num_kp + 1, -1, h, w))
        return sparse_deformed

    def normalize_point(self, R, t, c, raw):
        # R: B x 3 x 3
        # t: B x 3 x 1
        # c: B
        # raw: B x K x H x W x 3
        tmp = torch.einsum('bij,bchwjk->bchwik', R, raw.unsqueeze(5)) # B x K x H x W x 3 x 1
        tmp *= c.view(-1, 1, 1, 1, 1, 1) # B x K x H x W x 3 x 1
        tmp += t.unsqueeze(1).unsqueeze(2).unsqueeze(3) # B x 1 x 1 x 1 x 3 x 1
        normalized = tmp.squeeze(5) / 128 - 1 # B x K x H x W x 3

        return normalized

    def denormalize_point(self, R, t, c, normalized):
        # R: B x 3 x 3
        # t: B x 3 x 1
        # c: B
        # normalized: B x K x H x W x 3
        tmp = 128 * (normalized.unsqueeze(5) + 1) - t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        tmp = torch.einsum('bij,bchwjk->bchwik', R.inverse(), tmp)
        tmp = tmp / c.view(-1, 1, 1, 1, 1, 1)     
        denormalized = tmp.squeeze(5)  # B x K x H x W x 3
        return denormalized

    def get_segment_priors(self, mesh):
        priors = []
        for i, extractor in enumerate(self.motion_prior):
            segment = self.segments[i]
            segment_prior = extractor(mesh[:, segment].flatten(start_dim=-2))
            priors.append(segment_prior)
        return torch.cat(priors, dim=1)

    def get_segment_weights(self, input):
        weights = []
        prediction = self.hourglass(input)
        mask = self.mask(prediction)
        seg, weight_per_seg = mask[:, -(1 + len(self.segments)):], mask[:, :-(1 + len(self.segments))]
        seg = F.softmax(seg / self.T, dim=1)  # B x num_segs x H x W
        seg_0, seg = seg[:, [0]], seg[:, 1:]
        weights.append(seg_0)
        weight_per_seg = weight_per_seg.split(self.num_priors, dim=1)
        for i, weight in enumerate(weight_per_seg):
            # wegiht: B x num_prior x H x W
            weight = F.softmax(weight, dim=1)
            weights.append(weight * seg[:, [i]])
        weights = torch.cat(weights, dim=1) # B x 1 + num_kp x H x W
        return weights

    def forward(self, source_image, kp_driving, kp_source, driving_mesh_image=None):
        if self.scale_factor != 1:
            source_image = self.down(source_image)

        bs, _, h, w = source_image.shape

        out_dict = dict()
    
        v_driving = self.get_segment_priors(kp_driving['value']).view(bs, -1, 2)
        v_source = self.get_segment_priors(kp_source['value']).view(bs, -1, 2)

        kp_driving['value'] = v_source - v_driving
        
        sparse_motion = self.create_sparse_motions(source_image, kp_driving, kp_source)

        input = driving_mesh_image[:, [0]]
        mask = self.get_segment_weights(input)
        out_dict['mask'] = mask
        mask = mask.unsqueeze(2)
        sparse_motion = sparse_motion.permute(0, 1, 4, 2, 3)
        deformation = (sparse_motion * mask).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 1)   # B x H x W x 2
    
        out_dict['deformation'] = deformation

        return out_dict
