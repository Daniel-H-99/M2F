from torch import nn
import torch.nn.functional as F
from modules.util import kp2gaussian
import torch


class Discriminator(nn.Module):
    """
    Discriminator similar to Pix2Pix
    """

    def __init__(self, num_channels=3, block_expansion=64, num_blocks=4, max_features=512,
                 sn=False, use_kp=False, num_kp=10, kp_variance=0.01, **kwargs):
        super(Discriminator, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(
                DownBlock2d(num_channels + num_kp * use_kp + 1 if i == 0 else min(max_features, block_expansion * (2 ** i)),
                            min(max_features, block_expansion * (2 ** (i + 1))),
                            norm=(i != 0), kernel_size=4, pool=(i != num_blocks - 1), sn=sn))

        self.down_blocks = nn.ModuleList(down_blocks)
        self.conv = nn.Conv2d(self.down_blocks[-1].conv.out_channels, out_channels=1, kernel_size=1)
        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)
        self.use_kp = use_kp
        self.kp_variance = kp_variance

    def forward(self, x, kp=None):
        feature_maps = []
        out = x
        if self.use_kp:
            heatmap = kp2gaussian(kp, x.shape[2:], self.kp_variance)
            out = torch.cat([out, heatmap], dim=1)

        for down_block in self.down_blocks:
            feature_maps.append(down_block(out))
            out = feature_maps[-1]
        prediction_map = self.conv(out)

        return feature_maps, prediction_map


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale (scale) discriminator
    """

    def __init__(self, scales=(), **kwargs):
        super(MultiScaleDiscriminator, self).__init__()
        self.scales = scales
        discs = {}
        for scale in scales:
            discs[str(scale).replace('.', '-')] = Discriminator(**kwargs)
        self.discs = nn.ModuleDict(discs)

    def forward(self, x, kp=None):
        out_dict = {}
        for scale, disc in self.discs.items():
            scale = str(scale).replace('-', '.')
            key = 'prediction_' + scale
            feature_maps, prediction_map = disc(x[key], kp)
            out_dict['feature_maps_' + scale] = feature_maps
            out_dict['prediction_map_' + scale] = prediction_map
        return out_dict



class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class Encoder(nn.Module):
    def __init__(self, output_dim=64):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)
        
        self.fc = nn.Linear(512, output_dim)
        
    def forward(self, x):
        out = self.encoder(x).flatten(start_dim=1)
        out = self.fc(nn.ReLU()(out))
        return out
    
# class PriorEncoder(nn.Module):
#     def __init__(self, prior_dim=20, hidden_dim=96, embedding_dim=64, num_layers=2):
#         super(PriorEncoder, self).__init__()
#         self.prior_dim = prior_dim
#         self.lstm = nn.LSTM(prior_dim, hidden_dim, 2, dropout=0.2)
#         self.fc = nn.Linear(5 * hidden_dim, embedding_dim)
#     def forward(self, x):
#         # x: B x T x prior_dim
#         out = self.lstm(x)[0].flatten(start_dim=-2) # B x hidden_dim
#         out = self.fc(nn.ReLU()(out))   # B x hidden_dim
#         return out

class PriorEncoder_W5(nn.Module):
    def __init__(self, prior_dim=20, embedding_dim=64):
        super(PriorEncoder_W5, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(prior_dim, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=3)
        )
        self.prior_dim = prior_dim
        self.fc = nn.Linear(512, embedding_dim)
    def forward(self, x):
        # x: B x T x prior_dim
        x = x.transpose(1, 2)   # B x prior_dim x T
        # print(f'x shape: {x.shape}')
        out = self.layers(x) # B x 512 x 1
        out = self.fc(nn.ReLU()(out.flatten(1)))   # B x hidden_dim
        return out

class PriorEncoder_W3(nn.Module):
    def __init__(self, prior_dim=20, embedding_dim=64):
        super(PriorEncoder_W3, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(prior_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=3)
        )
        self.prior_dim = prior_dim
        self.fc = nn.Linear(512, embedding_dim)

    def forward(self, x):
        # x: B x T x prior_dim
        x = x.transpose(1, 2)   # B x prior_dim x T
        # print(f'x shape: {x.shape}')
        out = self.layers(x) # B x 512 x 1
        out = self.fc(nn.ReLU()(out.flatten(1)))   # B x hidden_dim
        return out

class LipDiscriminator(nn.Module):
    def __init__(self, prior_dim=20, embedding_dim=64, window=5):
        super(LipDiscriminator, self).__init__()
        self.audio_encoder = Encoder(output_dim=embedding_dim)
        self.prior_encoder = PriorEncoder_W5(prior_dim=prior_dim, embedding_dim=embedding_dim) if window == 5 else PriorEncoder_W3(prior_dim=prior_dim, embedding_dim=embedding_dim)
        self.loss_fn = nn.CosineEmbeddingLoss(reduction='none')

    def forward(self, audio, prior, label):
        # audio: B x :
        # prior: B x T x prior_dim
        # label: B (1 / -1)
        audio_embedding = self.audio_encoder(audio) # B x hidden_dim
        prior_embedding = self.prior_encoder(prior) # B x hidden_dim
        loss = self.loss_fn(audio_embedding, prior_embedding, label)
        return loss

class NoiseDiscriminator(nn.Module):
    def __init__(self, dim=20):
        super(NoiseDiscriminator, self).__init__()
        self.dim = dim
        self.layers = nn.Sequential(
            nn.Conv1d(self.dim, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=3)
        )
        self.fc = nn.Linear(512, 1)

    def forward(self, sequence):
        # sequence: B x window x dim
        x = sequence.transpose(1, 2)
        x = self.layers(x)
        x = self.fc(nn.ReLU()(x.flatten(1))).view(-1)
        return x
    
    def reality_loss(self, sequence):
        x = sequence.transpose(1, 2)
        x = self.layers(x)
        x = self.fc(nn.ReLU()(x.flatten(1))).view(-1)  # B
        loss = nn.BCEWithLogitsLoss(reduction='mean')(x, torch.ones_like(x))    # 1
        return loss

class NoiseGenerator(nn.Module):
    def __init__(self, noise=5e-3):
        super(NoiseGenerator, self).__init__()
        self.noise = noise
        self.noise = nn.Parameter(torch.Tensor([self.noise]).cuda())

    def forward(self, x):
        noise = self.noise * torch.randn_like(x).clamp(max=1)
        return x + noise

class DownBlock2d(nn.Module):
    """
    Simple block for processing video (encoder).
    """

    def __init__(self, in_features, out_features, norm=False, kernel_size=4, pool=False, sn=False):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size)

        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)

        if norm:
            self.norm = nn.InstanceNorm2d(out_features, affine=True)
        else:
            self.norm = None
        self.pool = pool

    def forward(self, x):
        out = x
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        out = F.leaky_relu(out, 0.2)
        if self.pool:
            out = F.avg_pool2d(out, (2, 2))
        return out
