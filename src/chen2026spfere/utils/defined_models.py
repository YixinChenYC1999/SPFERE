import torch
import torch.nn as nn
import torch.nn.functional as F

base_channels = 16

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def forward_debug(self, x):
        print(f"[Input]        {x.shape}")                      # [B, 3, 32, 32]
        x = self.pool(F.relu(self.conv1(x)))
        print(f"[Conv1+Pool]   {x.shape}")                      # [B, 6, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))
        print(f"[Conv2+Pool]   {x.shape}")                      # [B, 16, 5, 5]
        x = x.view(-1, 16 * 5 * 5)
        print(f"[Flatten]      {x.shape}")                      # [B, 400]
        x = F.relu(self.fc1(x))
        print(f"[FC1]          {x.shape}")                      # [B, 120]
        x = F.relu(self.fc2(x))
        print(f"[FC2]          {x.shape}")                      # [B, 84]
        x = self.fc3(x)
        print(f"[FC3 Output]   {x.shape}")                      # [B, 10]
        return x


class SimpleCNN_FM(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN_FM, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # input 1*28*28
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # Flatten change
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward_debug(self, x):
        print(f"[Input]        {x.shape}")                      # [B, 1, 28, 28]
        x = self.pool(F.relu(self.conv1(x)))
        print(f"[Conv1+Pool]   {x.shape}")                      # [B, 6, 12, 12]
        x = self.pool(F.relu(self.conv2(x)))
        print(f"[Conv2+Pool]   {x.shape}")                      # [B, 16, 4, 4]
        x = x.view(-1, 16 * 4 * 4)
        print(f"[Flatten]      {x.shape}")                      # [B, 256]
        x = F.relu(self.fc1(x))
        print(f"[FC1]          {x.shape}")                      # [B, 120]
        x = F.relu(self.fc2(x))
        print(f"[FC2]          {x.shape}")                      # [B, 84]
        x = self.fc3(x)
        print(f"[FC3 Output]   {x.shape}")                      # [B, 10]
        return x


class ShuffleCNN(nn.Module):
    def __init__(self, num_classes=10, base_channels=base_channels):
        super(ShuffleCNN, self).__init__()

        def conv_bn_relu(in_ch, out_ch, k=3, s=1, p=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.stage1 = conv_bn_relu(3, base_channels)
        # Shuffle Block for stage2
        branch_channels = base_channels // 2  # for splitting
        self.branch2 = nn.Sequential(
            nn.Conv2d(branch_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, stride=1, padding=1, groups=branch_channels, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.pool = nn.AvgPool2d(4)
        self.fc = nn.Linear(base_channels * 4 * 4, num_classes)  # output channels remain base_channels

    def forward(self, x):
        x = self.stage1(x)
        # Split input along channel dimension
        x1, x2 = x.chunk(2, dim=1)
        x1 = F.max_pool2d(x1, 2)
        out2 = self.branch2(x2)
        # Concatenate and shuffle
        out = torch.cat((x1, out2), dim=1)
        out = self.channel_shuffle(out, groups=2)
        x = self.pool(out)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    @staticmethod
    def channel_shuffle(x, groups):
        B, C, H, W = x.size()
        x = x.view(B, groups, C // groups, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x.view(B, C, H, W)
    
    def forward_debug(self, x):
        print(f"[Input]        {x.shape}")
        x = self.stage1(x)
        print(f"[Stage1]       {x.shape}")
        x1, x2 = x.chunk(2, dim=1)
        print(f"[Chunk x1/x2]  {x1.shape} / {x2.shape}")
        x1 = F.max_pool2d(x1, 2)
        print(f"[x1 Pooled]    {x1.shape}")
        out2 = self.branch2(x2)
        print(f"[Branch2 out2] {out2.shape}")
        out = torch.cat((x1, out2), dim=1)
        print(f"[Concat]       {out.shape}")
        out = self.channel_shuffle(out, groups=2)
        print(f"[Shuffled]     {out.shape}")
        x = self.pool(out)
        print(f"[Pooled]       {x.shape}")
        x = torch.flatten(x, 1)
        print(f"[Flattened]    {x.shape}")
        x = self.fc(x)
        print(f"[Output]       {x.shape}")
        return x


class EfficientCNN(nn.Module):
    def __init__(self, num_classes=10, base_channels=base_channels):
        super(EfficientCNN, self).__init__()

        def conv_bn_relu(in_ch, out_ch, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        def depthwise_block(in_ch, hidden_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, hidden_ch, kernel_size=1, bias=False),
                # nn.BatchNorm2d(hidden_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=1, groups=hidden_ch, bias=False),
                # nn.BatchNorm2d(hidden_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )
            
        self.stage1 = conv_bn_relu(3, base_channels)
        self.stage2 = depthwise_block(base_channels, base_channels, base_channels)        # DW Block (pw+dw+pw)
        self.pool = nn.AvgPool2d(4)
        self.fc = nn.Linear(base_channels * 4 * 4, num_classes)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def forward_debug(self, x):
        print(f"[Input]        {x.shape}")                      # [B, 3, 32, 32]
        x = self.stage1(x)
        print(f"[Stage1]       {x.shape}")                      # [B, base, 32, 32]
        x = self.stage2(x)
        print(f"[Stage2]       {x.shape}")                      # [B, base*2, 16, 16]
        x = self.pool(x)
        print(f"[Pool]         {x.shape}")                      # [B, base*2, 4, 4]
        x = torch.flatten(x, 1)
        print(f"[Flatten]      {x.shape}")                      # [B, base*2 * 4 * 4]
        x = self.fc(x)
        print(f"[FC Output]    {x.shape}")                      # [B, num_classes]
        return x


class EfficientCNN_DP(nn.Module):
    def __init__(self, dropout_p=0, num_classes=10, base_channels=base_channels):
        super(EfficientCNN_DP, self).__init__()

        def conv_bn_relu(in_ch, out_ch, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        def depthwise_block(in_ch, hidden_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, hidden_ch, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=1, groups=hidden_ch, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )
            
        self.stage1 = conv_bn_relu(3, base_channels)
        self.stage2 = depthwise_block(base_channels, base_channels, base_channels)  # DW Block
        self.pool = nn.AvgPool2d(4)

        self.fc = nn.Linear(base_channels * 4 * 4, 32)   # 32
        self.dropout = nn.Dropout(p=dropout_p)            # dropout
        self.fc_out = nn.Linear(32, num_classes)         # output

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        return x