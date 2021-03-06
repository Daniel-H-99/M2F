from torch import nn
import torch
import torch.nn.functional as F
from modules.util import AntiAliasInterpolation2d, make_coordinate_grid, MASK_IDX, LIP_IDX, OVAL_IDX, CONTOUR_IDX, ROI_IDX
from torchvision import models
import numpy as np
from torch.autograd import grad


class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
    """
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict


class Transform:
    """
    Random tps transformation for equivariance constraints. See Sec 3.3
    """
    def __init__(self, bs, **kwargs):
        noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid((kwargs['points_tps'], kwargs['points_tps']), type=noise.type())
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(mean=0,
                                               std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 2]))
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:], type=frame.type()).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, padding_mode="reflection")

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)

        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = torch.abs(distances).sum(-1)

            result = distances ** 2
            result = result * torch.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result

        return transformed

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        grad_x = grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_y = grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian


def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}


class MeshGeneratorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, generator, discriminator, train_params):
        super(MeshGeneratorFullModel, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = train_params['scales']
        self.disc_scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels + 1)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()

    def forward(self, x):
        # print("input shape: {}".format(x['source'].shape))
        # print("input shape - mesh: {}".format(x['source_mesh']['mesh'].shape))
        # print("input shape - R: {}".format(x['source_mesh']['R'].shape))
        # print("input shape - t: {}".format(x['source_mesh']['t'].shape))
        # print("input shape - c: {}".format(x['source_mesh']['c'].shape))

        kp_source = self.preprocess_mesh(x['source_mesh'])
        kp_driving = self.preprocess_mesh(x['driving_mesh'])

        generated = self.generator(x['source'], kp_source=kp_source, kp_driving=kp_driving, driving_mesh_image=x['driving_mesh_image'], driving_image=x['driving'])
        generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})

        loss_values = {}

        pyramide_real = self.pyramid(torch.cat([x['real'], x['real_mesh_image'].cuda()[:, [0]]], dim=1))
        pyramide_generated = self.pyramid(torch.cat([generated['prediction'], x['driving_mesh_image'].cuda()[:, [0]]], dim=1))

        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)][:, :-1])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)][:, :-1])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
                loss_values['perceptual'] = value_total

        if self.loss_weights['generator_gan'] != 0:
            discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_driving))
            discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_driving))
            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                value_total += self.loss_weights['generator_gan'] * value
            loss_values['gen_gan'] = value_total

            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = torch.abs(a - b).mean()
                        value_total += self.loss_weights['feature_matching'][i] * value
                    loss_values['feature_matching'] = value_total

        # calc landmark motion error
        if self.loss_weights['motion'] != 0:
            roi = ROI_IDX
            mask = MASK_IDX
            boundary = OVAL_IDX
            bg_mask = (x['driving_mesh_image'] == 0)[:, 0] # B x H x W
            deformation = generated['deformation']  # B x H x W x 2 
            driving_mesh = x['driving_mesh']['mesh'].cuda()[:, :, None, :2] # B x K x 1 x 2
            driving_mesh_roi = driving_mesh[:, roi]
            driving_mesh_boundary = driving_mesh[:, boundary]
            driving_mesh_normalized = x['driving_mesh']['normed_mesh'].cuda()
            motion = F.grid_sample(deformation.permute(0, 3, 1, 2), driving_mesh).squeeze(3).permute(0, 2, 1).flatten(start_dim=1)   # B x K * 2
            motion_GT = x['source_mesh']['mesh'].cuda()[:, :, :2].flatten(start_dim=1) # B x K * 2
            motion_roi = F.grid_sample(deformation.permute(0, 3, 1, 2), driving_mesh_roi).squeeze(3).permute(0, 2, 1).flatten(start_dim=1) 
            motion_roi_GT = x['source_mesh']['mesh'].cuda()[:, roi, :2].flatten(start_dim=1) # B x K * 2
            motion_boundary = F.grid_sample(deformation.permute(0, 3, 1, 2), driving_mesh_boundary).squeeze(3).permute(0, 2, 1).flatten(start_dim=1) 
            motion_boundary_GT = x['source_mesh']['mesh'].cuda()[:, boundary, :2].flatten(start_dim=1) # B x K * 2
            background_loss = self.loss_weights['background'] * F.l1_loss(generated['mask'][:, 0] * bg_mask, torch.ones_like(bg_mask).float() * bg_mask)
            motion_loss = self.loss_weights['motion'] * F.mse_loss(motion, motion_GT)
            motion_roi_loss = self.loss_weights['motion_roi'] * F.l1_loss(motion_roi, motion_roi_GT)
            motion_boundary_loss = self.loss_weights['motion_boundary'] * F.l1_loss(motion_boundary, motion_boundary_GT)
            loss_values['bg'] = background_loss
            loss_values['motion'] = motion_loss
            loss_values['motion_roi'] = motion_roi_loss
            loss_values['motion_boundary'] = motion_boundary_loss
        return loss_values, generated

    def preprocess_mesh(self, mesh):
        roi = ROI_IDX
        res = mesh
        res['value'] = mesh['normed_mesh'][:, :, :2]
        # res['jacobian'] = (mesh['R'].inverse()[:, :2, :2] / mesh['c'].unsqueeze(1).unsqueeze(2)).unsqueeze(1).repeat(1, res['value'].size(1), 1, 1)

        # print("jacobian shape: {}".format(res['jacobian'].shape))
        return res

class MeshDiscriminatorFullModel(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, generator, discriminator, train_params):
        super(MeshDiscriminatorFullModel, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels + 1)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

    def forward(self, x, generated):
        pyramide_real = self.pyramid(torch.cat([x['real'], x['real_mesh_image'].cuda()[:, [0]]], dim=1))
        pyramide_generated = self.pyramid(torch.cat([generated['prediction'].detach(), x['driving_mesh_image'].cuda()[:, [0]]], dim=1))

        kp_driving = generated['kp_driving']
        discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_driving))
        discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_driving))

        loss_values = {}
        value_total = 0
        for scale in self.scales:
            key = 'prediction_map_%s' % scale
            value = (1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2
            value_total += self.loss_weights['discriminator_gan'] * value.mean()
        loss_values['disc_gan'] = value_total

        return loss_values

