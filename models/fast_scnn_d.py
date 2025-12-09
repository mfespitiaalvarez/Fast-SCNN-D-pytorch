import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['FastSCNND']

# --- New Module 1: Pseudo-HHA Generator ---
class GeometryFeatureGenerator(nn.Module):
    """
    Computes 'Pseudo-HHA' features on the GPU in real-time.
    Input:  [B, 1, H, W] (Raw Normalized Disparity)
    Output: [B, 3, H, W] (Depth, Normal_X, Normal_Y)
    """
    def __init__(self):
        super(GeometryFeatureGenerator, self).__init__()
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        
        # Standard Sobel Kernels
        sobel_kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        
        self.sobel_x.weight = nn.Parameter(sobel_kernel_x, requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_kernel_y, requires_grad=False)

    def forward(self, x):
        grad_x = self.sobel_x(x)
        grad_y = self.sobel_y(x)
        
        # Calculate approximate Surface Normals
        magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        norm_x = grad_x / magnitude
        norm_y = grad_y / magnitude
        
        return torch.cat([x, norm_x, norm_y], dim=1)

# --- New Module 2: Gated Fusion ---
class GatedFusion(nn.Module):
    """
    Learns to dynamically weight RGB vs Geometry features.
    """
    def __init__(self, in_channels):
        super(GatedFusion, self).__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
    
    def forward(self, x_rgb, x_depth):
        cat_feats = torch.cat([x_rgb, x_depth], dim=1)
        alpha = self.gate_conv(cat_feats)
        return (x_rgb * alpha) + (x_depth * (1 - alpha))

# --- New Module 3: Dual-Stream LDS ---
class DualLearningToDownsample(nn.Module):
    """
    Processes RGB and HHA in separate streams before fusion.
    """
    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64, **kwargs):
        super(DualLearningToDownsample, self).__init__()
        
        # Module to convert 1-ch Disparity to 3-ch Geometry
        self.geo_generator = GeometryFeatureGenerator()
        
        # Stream A: RGB
        self.rgb_conv = _ConvBNReLU(3, dw_channels1, 3, 2)
        self.rgb_dsconv1 = _DSConv(dw_channels1, dw_channels2, 2)
        self.rgb_dsconv2 = _DSConv(dw_channels2, out_channels, 2)
        
        # Stream B: Geometry (Input is 3 channels from generator)
        self.depth_conv = _ConvBNReLU(3, dw_channels1, 3, 2)
        self.depth_dsconv1 = _DSConv(dw_channels1, dw_channels2, 2)
        self.depth_dsconv2 = _DSConv(dw_channels2, out_channels, 2)
        
        # Fusion
        self.fusion = GatedFusion(out_channels)

    def forward(self, x):
        # x is [Batch, 4, H, W]
        x_rgb = x[:, :3, :, :]
        x_raw_disp = x[:, 3:, :, :]
        
        # 1. Expand Disparity to HHA
        x_geo = self.geo_generator(x_raw_disp)
        
        # 2. RGB Stream
        r = self.rgb_conv(x_rgb)
        r = self.rgb_dsconv1(r)
        r = self.rgb_dsconv2(r)
        
        # 3. Geometry Stream
        d = self.depth_conv(x_geo)
        d = self.depth_dsconv1(d)
        d = self.depth_dsconv2(d)
        
        # 4. Fuse
        return self.fusion(r, d)

# --- Modified Main Class ---
class FastSCNND(nn.Module):
    def __init__(self, num_classes, aux=False, **kwargs):
        super(FastSCNND, self).__init__()
        self.aux = aux
        
        # Replaced standard LDS with Dual-Stream
        self.learning_to_downsample = DualLearningToDownsample(32, 48, 64)
        
        # Standard Backbone components (unchanged)
        self.global_feature_extractor = GlobalFeatureExtractor(64, [64, 96, 128], 128, 6, [3, 3, 3])
        self.feature_fusion = FeatureFusionModule(64, 128, 128)
        self.classifier = Classifer(128, num_classes)
        
        if self.aux:
            self.auxlayer = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(32, num_classes, 1)
            )

    def forward(self, x):
        size = x.size()[2:]
        # x is now [B, 4, H, W]
        
        # 1. Dual Stream + Fusion
        fused_features = self.learning_to_downsample(x)
        
        # 2. Shared Backbone
        global_context = self.global_feature_extractor(fused_features)
        
        # 3. Skip Connection Fusion
        # Fused features connect here!
        x = self.feature_fusion(fused_features, global_context)
        
        x = self.classifier(x)
        outputs = []
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(fused_features)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)



class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _DSConv(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _DWConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class LinearBottleneck(nn.Module):
    """LinearBottleneck used in MobileNetV2"""

    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            # pw
            _ConvBNReLU(in_channels, in_channels * t, 1),
            # dw
            _DWConv(in_channels * t, in_channels * t, stride),
            # pw-linear
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out


class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.out = _ConvBNReLU(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x


class LearningToDownsample(nn.Module):
    """Learning to downsample module"""

    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64, **kwargs):
        super(LearningToDownsample, self).__init__()
        self.conv = _ConvBNReLU(3, dw_channels1, 3, 2)
        self.dsconv1 = _DSConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = _DSConv(dw_channels2, out_channels, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class GlobalFeatureExtractor(nn.Module):
    """Global feature extractor module"""

    def __init__(self, in_channels=64, block_channels=(64, 96, 128),
                 out_channels=128, t=6, num_blocks=(3, 3, 3), **kwargs):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 2)
        self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 2)
        self.bottleneck3 = self._make_layer(LinearBottleneck, block_channels[1], block_channels[2], num_blocks[2], t, 1)
        self.ppm = PyramidPooling(block_channels[2], out_channels)

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = []
        layers.append(block(inplanes, planes, t, stride))
        for i in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.ppm(x)
        return x


class FeatureFusionModule(nn.Module):
    """Feature fusion module"""

    def __init__(self, highter_in_channels, lower_in_channels, out_channels, scale_factor=4, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = _DWConv(lower_in_channels, out_channels, 1)
        self.conv_lower_res = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.conv_higher_res = nn.Sequential(
            nn.Conv2d(highter_in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(True)

    def forward(self, higher_res_feature, lower_res_feature):
        lower_res_feature = F.interpolate(lower_res_feature, scale_factor=4, mode='bilinear', align_corners=True)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)


class Classifer(nn.Module):
    """Classifer"""

    def __init__(self, dw_channels, num_classes, stride=1, **kwargs):
        super(Classifer, self).__init__()
        self.dsconv1 = _DSConv(dw_channels, dw_channels, stride)
        self.dsconv2 = _DSConv(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(dw_channels, num_classes, 1)
        )

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x


def get_fast_scnn(dataset='citys', pretrained=False, root='./weights', map_cpu=False, **kwargs):
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
        'citys_d': 'citys',  # Disparity version uses same weights naming
    }
    from data_loader import datasets
    # Handle citys_d dataset name
    dataset_key = dataset if dataset in datasets else 'citys'
    model = FastSCNND(datasets[dataset_key].NUM_CLASS, **kwargs)
    if pretrained:
        if(map_cpu):
            model.load_state_dict(torch.load(os.path.join(root, 'fast_scnn_%s.pth' % acronyms[dataset]), map_location='cpu'))
        else:
            model.load_state_dict(torch.load(os.path.join(root, 'fast_scnn_%s.pth' % acronyms[dataset])))
    return model


if __name__ == '__main__':
    # Test with 4 channels (3 RGB + 1 Depth)
    img = torch.randn(2, 4, 1024, 2048) 
    model = get_fast_scnn('citys', pretrained=False)
    
    # Move to GPU if available to test Cuda kernels
    if torch.cuda.is_available():
        model = model.cuda()
        img = img.cuda()
        
    outputs = model(img)
    print("Output shape:", outputs[0].shape) # Should be [2, 19, 1024, 2048]