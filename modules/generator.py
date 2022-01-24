import torch
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
from modules.dense_motion import MeshDenseMotionNetwork

class MeshOcclusionAwareGenerator(nn.Module):
    """
    Generator that given source image and and keypoints try to transform image according to movement trajectories
    induced by keypoints. Generator follows Johnson architecture.
    """

    def __init__(self, num_channels, num_kp, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, estimate_occlusion_map=False, dense_motion_params=None, estimate_jacobian=False):
        super(MeshOcclusionAwareGenerator, self).__init__()

        if dense_motion_params is not None:
            self.dense_motion_network = MeshDenseMotionNetwork(num_kp=num_kp, num_channels=num_channels,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None

        self.first = SameBlock2d(num_channels + 1, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i))) * 2
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.estimate_occlusion_map = estimate_occlusion_map
        self.num_channels = num_channels

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear')
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation)

    def forward(self, source_image, kp_driving, kp_source, driving_mesh_image=None, driving_image=None):
        # Encoding (downsampling) part
        # Transforming feature representation according to deformation and occlusion
        out = source_image
        output_dict = {}
        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(source_image=source_image, kp_driving=kp_driving,
                                                     kp_source=kp_source, driving_mesh_image=driving_mesh_image)
            output_dict['kp_source'] = kp_source
            output_dict['kp_driving'] = kp_driving
            output_dict['mask'] = dense_motion['mask']  # B x K x H x W
            output_dict['deformation'] = dense_motion['deformation']
            
            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']
                output_dict['occlusion_map'] = occlusion_map
            else:
                occlusion_map = None

            deformation = dense_motion['deformation']
            out = self.deform_input(out, deformation)
            deformed_source = out
            if occlusion_map is not None:
                if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear')
                out = out * occlusion_map

            if driving_image is not None:
                # driving_mask = output_dict['mask'][:, [0]].detach()
                # driving_mask *= (driving_mesh_image[:, [0]] == 0)
                driving_mask = (driving_mesh_image[:, [0]] == 0).byte()
                out = out * (1 - driving_mask) + driving_image * driving_mask

            output_dict["deformed"] = out
        
        out = torch.cat([out, F.interpolate(driving_mesh_image[:, [0]], out.shape[2:])], dim=1)
        out = self.first(out)

        features = []
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
            features.append(out)

        # Decoding part
        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            out = torch.cat([out, features.pop()], dim=1)
            out = self.up_blocks[i](out)
        out = self.final(out)
        out = F.sigmoid(out)

 
        if occlusion_map is not None:
            if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
                occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear')
            out = out * occlusion_map + source_image * (1 - occlusion_map)

        output_dict["prediction"] = out

        return output_dict
