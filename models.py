import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.dropout = nn.Dropout3d(p=0.3)

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)

        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck3D, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.conv = nn.Conv3d(1, in_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, mask, size):
        mask_resized = F.interpolate(mask, size=size, mode='trilinear', align_corners=False)
        attention = self.sigmoid(self.bn(self.conv(mask_resized)))
        return attention


class ResNet3D(nn.Module):
    def __init__(self, block, layers, num_classes=4, attention=False):
        super(ResNet3D, self).__init__()
        self.inplanes_ct = 32
        self.inplanes_pet = 32
        self.attention = attention
        if attention:
            self.attention_ct = AttentionModule(32)
            self.attention_pet = AttentionModule(32)
        self.conv1_ct = nn.Conv3d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_ct = nn.BatchNorm3d(32)
        self.conv1_pet = nn.Conv3d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_pet = nn.BatchNorm3d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1_ct = self._make_layer(block, 32, layers[0], inplanes='ct')
        self.layer1_pet = self._make_layer(block, 32, layers[0], inplanes='pet')
        self.layer2_ct = self._make_layer(block, 64, layers[1], stride=2, inplanes='ct')
        self.layer2_pet = self._make_layer(block, 64, layers[1], stride=2, inplanes='pet')
        self.layer3_ct = self._make_layer(block, 128, layers[2], stride=2, inplanes='ct')
        self.layer3_pet = self._make_layer(block, 128, layers[2], stride=2, inplanes='pet')
        self.layer4_ct = self._make_layer(block, 256, layers[3], stride=2, inplanes='ct')
        self.layer4_pet = self._make_layer(block, 256, layers[3], stride=2, inplanes='pet')

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(2 * 256 * block.expansion, num_classes)
        self.b = nn.Parameter(torch.tensor(10.0))
        self.dropout = nn.Dropout(p=0.2)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, inplanes='ct'):
        downsample = None
        input_planes = getattr(self, f'inplanes_{inplanes}')
        if stride != 1 or input_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(input_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(input_planes, planes, stride, downsample))
        setattr(self, f'inplanes_{inplanes}', planes * block.expansion)
        for _ in range(1, blocks):
            layers.append(block(getattr(self, f'inplanes_{inplanes}'), planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        ct = x[:, 0:1, :, :, :]
        pet = x[:, 1:2, :, :, :]
        seg = x[:, 2:3, :, :, :]
        if self.attention:
            seg_modified = seg * (self.b) + 1
            ct = ct * seg_modified
            pet = pet * seg_modified
        # CT
        ct = self.conv1_ct(ct)
        ct = self.bn1_ct(ct)
        ct = self.relu(ct)
        ct = self.maxpool(ct)
        ct = self.layer1_ct(ct)
        ct = self.layer2_ct(ct)
        ct = self.layer3_ct(ct)
        ct = self.layer4_ct(ct)
        ct = self.avgpool(ct)
        ct = torch.flatten(ct, 1)

        # PET
        pet = self.conv1_pet(pet)
        pet = self.bn1_pet(pet)
        pet = self.relu(pet)
        pet = self.maxpool(pet)
        pet = self.layer1_pet(pet)
        pet = self.layer2_pet(pet)
        pet = self.layer3_pet(pet)
        pet = self.layer4_pet(pet)
        pet = self.avgpool(pet)
        pet = torch.flatten(pet, 1)
        x = torch.cat([ct, pet], dim=1)
        x = self.fc(x)
        return x



class RadioLOGIC(torch.nn.Module):

    def __init__(self):
        super(RadioLOGIC, self).__init__()
        self.bert = RobertaModel.from_pretrained(
            '/home/x.liang/MyProject/survivalPrediction/radiobert_BigDataset_epoch10',
            add_pooling_layer=False)

    def forward(self, input_id=None, attention_mask=None):  # , return_dict=None

        outputs = self.bert(input_ids=input_id, attention_mask=attention_mask, return_dict=False)
        sequence_output = outputs[0]
        pooler = sequence_output[:, 0]

        return pooler


class ReportNet(nn.Module):
    def __init__(self):
        super(ReportNet, self).__init__()
        self.RadioLOGIC = RadioLOGIC()
        for param in self.RadioLOGIC.parameters():
            param.requires_grad = True
        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, 128)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_id=None, attention_mask=None):
        x = self.RadioLOGIC(input_id, attention_mask)
        x = torch.relu(self.fc1(x))
        intermediate_features = x.detach()
        x = self.fc2(x)
        return x, intermediate_features


class ClinicalNet(nn.Module):
    def __init__(self):
        super(ClinicalNet, self).__init__()
        self.fc1 = nn.Linear(25 * 10, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MIAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.,
            proj_drop=0.,
    ):
        super(MIAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm2 = nn.LayerNorm(128)
        self.mlp = Mlp(in_features=128, hidden_features=128, out_features=128,
                       act_layer=nn.GELU, drop=0.1)

    def forward(self, image, report):
        image_res = image
        report_res = report
        B, N_image, C = image.shape
        _, N_report, _ = report.shape

        # Process image and report through the QKV network
        qkv_image = self.qkv(image).reshape(B, N_image, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_image, k_image, v_image = qkv_image[0], qkv_image[1], qkv_image[2]

        qkv_report = self.qkv(report).reshape(B, N_report, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                           4)
        q_report, k_report, v_report = qkv_report[0], qkv_report[1], qkv_report[2]

        # Cross-attention calculations between image and report
        attn_image = (q_image @ k_report.transpose(-2, -1)) * self.scale
        attn_report = (q_report @ k_image.transpose(-2, -1)) * self.scale

        # Applying softmax and dropout
        attn_image = self.attn_drop(attn_image.softmax(dim=-1))
        attn_report = self.attn_drop(attn_report.softmax(dim=-1))

        # Attention outputs
        attn_image_x = (attn_image @ v_report).transpose(1, 2).reshape(B, N_image, C)
        attn_report_x = (attn_report @ v_image).transpose(1, 2).reshape(B, N_report, C)

        # Projection and final processing
        attn_image_x = self.proj_drop(self.proj(attn_image_x))
        attn_report_x = self.proj_drop(self.proj(attn_report_x))

        image = image * attn_image_x
        report = report * attn_report_x

        image = image_res + image
        report = report_res + report

        return image + report


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.fc = nn.Linear(128 * 3, 128)
        self.classifier = nn.Linear(128, 4)
        self.prompt = nn.Parameter(torch.tensor([[1.0, 0.0]]))
        self.prompt.requires_grad = False
        self.attn_image_report = MIAttention(dim=128,
                                             num_heads=4,
                                             qkv_bias=True,
                                             attn_drop=0,
                                             proj_drop=0,
                                             )
        self.pub = nn.Sequential(
            nn.Linear(128 * 3, 128),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.2),
            nn.Linear(128, 128),
        )
        self.attn_rad_clin = MIAttention(dim=128,
                                         num_heads=4,
                                         qkv_bias=True,
                                         attn_drop=0,
                                         proj_drop=0,
                                         )
        self.norm1 = nn.LayerNorm(128)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, image_features, report_features, clin_features):
        pub_image_report = self.attn_image_report(image_features.unsqueeze(1), report_features.unsqueeze(1)).squeeze(1)
        radology = self.pub(torch.cat((image_features, pub_image_report, report_features), dim=1))
        pub_rad_clin = self.attn_rad_clin(radology.unsqueeze(1), clin_features.unsqueeze(1)).squeeze(1)
        combined_features = torch.cat((radology, pub_rad_clin, clin_features), dim=1)
        combined_features = torch.relu(self.fc(combined_features))
        outputs = self.classifier(combined_features)
        return outputs, 0


class MOPS(nn.Module):
    def __init__(self):
        super(MOPS, self).__init__()
        self.image_branch = ResNet3D(BasicBlock3D, [2, 2, 2, 2], num_classes=128, attention=True)
        self.report_branch = ReportNet()
        self.clin_branch = ClinicalNet()
        self.classifier = Classifier()  

    def forward(self, image, input_id=None, attention_mask=None, clinical=None, prompt=None):
        image_features = self.image_branch(image)
        report_features, intermediate_vector_features = self.report_branch(input_id, attention_mask)
        clin_features = self.clin_branch(clinical.view(clinical.size(0), -1))
        outputs = self.classifier(image_features, report_features, clin_features)
        return outputs