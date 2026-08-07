from torch import nn, Tensor

class BaseModel(nn.Module):
    def __init__(self) -> None:
        super(BaseModel, self).__init__()
        self.layers = nn.ModuleList([])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            try:
                if isinstance(layer, nn.Linear) or (isinstance(layer[0], nn.Dropout) and (len(layer) == 3)):
                    x = x.view(x.size(0), -1)
            except TypeError:
                pass
            x = layer(x)
        return x
    
    def set_layers_client(self, layers: nn.ModuleList, partition_point: int) -> None:
        for i in range(partition_point+1):
            self.layers.append(layers[i])
        
    def set_layers_server(self, layers: nn.ModuleList, partition_point: int) -> None:
        for i in range(partition_point+1, len(layers)):
            self.layers.append(layers[i])
    
    def set_layers(self, layers: nn.ModuleList) -> None:
        for i in range(len(layers)):
            self.layers.append(layers[i])

class ResidualBlock(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            stride: int = 1, 
            downsample: nn.Sequential | None = None
            ) -> None:
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        
        out = self.relu(out)
        return out
    
class BottleneckBlock(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            bottleneck_channels: int, 
            out_channels: int, 
            stride: int = 1, 
            downsample: nn.Sequential | None = None
            ) -> None:
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, bottleneck_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(bottleneck_channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(bottleneck_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        
        out = self.relu(out)
        return out
    
class ResNet(BaseModel):
    def __init__(self, num_blocks: list[int], has_bottleneck: bool = False) -> None:
        super(ResNet, self).__init__()
        self.input_planes = 64
        self.has_bottleneck = has_bottleneck
        self.layer_dims = [64, 128, 256, 512]
        self.bottleneck_dims = [256, 512, 1024, 2048]
        self.out_channels = 10

        self.layers.append(nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ))
        # conv1
        self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        # conv2_x
        self.layers.append(self._make_layer(self.layer_dims[0], num_blocks[0], self.bottleneck_dims[0]))
        # conv3_x
        self.layers.append(self._make_layer(self.layer_dims[1], num_blocks[1], self.bottleneck_dims[1], stride=2))
        # conv4_x
        self.layers.append(self._make_layer(self.layer_dims[2], num_blocks[2], self.bottleneck_dims[2], stride=2))
        # conv5_x
        self.layers.append(self._make_layer(self.layer_dims[3], num_blocks[3], self.bottleneck_dims[3], stride=2))
        self.layers.append(nn.AvgPool2d(7, stride=1))
        self.layers.append(nn.Linear(self.layer_dims[3], self.out_channels))
        
    def _make_layer(self, planes: int, num_blocks: int, bottleneck_planes: int, stride: int = 1) -> nn.Sequential:
        downsample = None

        if stride != 1 or planes != self.input_planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.input_planes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes)
            )
        temp_layers = []
        if self.has_bottleneck:
            temp_layers.append(BottleneckBlock(self.input_planes, bottleneck_planes, planes, stride, downsample))
        else:
            temp_layers.append(ResidualBlock(self.input_planes, planes, stride, downsample))
        self.input_planes = planes

        for i in range(1, num_blocks):
            if self.has_bottleneck:
                temp_layers.append(BottleneckBlock(self.input_planes, bottleneck_planes, planes))
            else:
                temp_layers.append(ResidualBlock(self.input_planes, planes))
        
        return nn.Sequential(*temp_layers)

class VGG(BaseModel):
    def __init__(self, num_classes: int = 10, num_blocks: list[int] = [1, 2, 3, 3, 3], change_dim: bool = False) -> None:
        super(VGG, self).__init__()
        self.layer_dims = [3, 64, 128, 256, 512, 512]
        # Block 0
        self.layers.append(nn.Sequential(
            nn.Conv2d(self.layer_dims[0], self.layer_dims[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        )
        # Block 1
        self.layers.append(self._make_layer(1, num_blocks[0], change_dim))
        # Block 2
        self.layers.append(self._make_layer(2, num_blocks[1], change_dim))
        # Block 3
        self.layers.append(self._make_layer(3, num_blocks[2]))
        # Block 4
        self.layers.append(self._make_layer(4, num_blocks[3]))
        # Block 5
        self.layers.append(self._make_layer(5, num_blocks[4]))
        # Block 6
        self.layers.append(nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*512, 4096),
            nn.ReLU())
        )
        # Block 7
        self.layers.append(nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Linear(4096, num_classes))
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            if isinstance(layer[0], nn.Dropout) and (len(layer) == 3):
                x = x.view(x.size(0), -1)
            x = layer(x)
        return x

    def _make_layer(self, planes_index: int, num_blocks: int, change_dim: bool = False) -> nn.Sequential:
        in_planes = self.layer_dims[planes_index - 1]
        out_planes = self.layer_dims[planes_index]

        # VGG11 only
        if num_blocks == 0:
            return nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
        elif (num_blocks == 1) and change_dim:
            return nn.Sequential(
                    nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(out_planes),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )

        temp_layers = []

        for i in range(num_blocks):
            if i == num_blocks-1:
                temp_layers.append(nn.Sequential(
                    nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(out_planes),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                ))
            elif i == 0:
                temp_layers.append(nn.Sequential(
                    nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(out_planes),
                    nn.ReLU()
                ))
            else:
                temp_layers.append(nn.Sequential(
                    nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(out_planes),
                    nn.ReLU()
                ))
                

        return nn.Sequential(*temp_layers)
    
def get_model_by_name(model_name: str = "ResNet18") -> ResNet:
    match model_name:
        case "ResNet18":
            return ResNet(num_blocks=[2, 2, 2, 2])
        case "ResNet34":
            return ResNet(num_blocks=[3, 4, 6, 3])
        case "ResNet50":
            return ResNet(num_blocks=[3, 4, 6, 3], has_bottleneck=True)
        case "ResNet101":
            return ResNet(num_blocks=[3, 4, 23, 3], has_bottleneck=True)
        case "ResNet152":
            return ResNet(num_blocks=[3, 8, 36, 3], has_bottleneck=True)
        case "VGG11":
            return VGG(num_classes=10, num_blocks=[0, 1, 2, 2, 2], change_dim=True)
        case "VGG16":
            return VGG(num_classes=10, num_blocks=[1, 2, 3, 3, 3])
        case "VGG19":
            return VGG(num_classes=10, num_blocks=[1, 2, 4, 4, 4])
        case _:
            raise Exception(f"'{model_name}' is not a valid model name")