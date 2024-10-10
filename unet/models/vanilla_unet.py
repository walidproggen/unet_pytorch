import torch
from unet.models.unet_layers import ConvBlock, UpConvBlock


class Unet(torch.nn.Module):
    def __init__(self, input_dim=1, nr_classes=1, conv_dim=64, kernel_size=3, padding='same'):
        super().__init__()
        self.padding = padding
        self.conv_dim = conv_dim
        self.nr_classes = nr_classes
        self.kernel_size = kernel_size

        # Encoder layers
        self.x1 = torch.nn.Sequential(
            ConvBlock(input_dim, conv_dim, kernel_size, padding)
        )
        self.x2 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(conv_dim, conv_dim * 2, kernel_size, padding)
        )
        self.x3 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(conv_dim * 2, conv_dim * 4, kernel_size, padding)
        )
        self.x4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(conv_dim * 4, conv_dim * 8, kernel_size, padding)
        )
        self.x5 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(conv_dim * 8, conv_dim * 16, kernel_size, padding)
        )

        # Decoder Layers
        self.x6 = UpConvBlock(conv_dim * 16, padding)
        self.x6_conv = ConvBlock(conv_dim * 16, conv_dim * 8, kernel_size, padding)
        self.x7 = UpConvBlock(conv_dim * 8, padding)
        self.x7_conv = ConvBlock(conv_dim * 8, conv_dim * 4, kernel_size, padding)
        self.x8 = UpConvBlock(conv_dim * 4, padding)
        self.x8_conv = ConvBlock(conv_dim * 4, conv_dim * 2, kernel_size, padding)
        self.x9 = UpConvBlock(conv_dim * 2, padding)
        self.x9_conv = ConvBlock(conv_dim * 2, conv_dim, kernel_size, padding)

        # Final 1x1 conv-layer
        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(conv_dim, nr_classes, kernel_size=1, padding=padding, bias=False),
            torch.nn.BatchNorm2d(nr_classes),
            #torch.nn.Sigmoid() -> We use BCEWithLogitsLoss() instead
        )

    def forward(self, x):
        # Encode
        x1 = self.x1(x)
        x2 = self.x2(x1)
        x3 = self.x3(x2)
        x4 = self.x4(x3)
        x5 = self.x5(x4)

        # Decode
        x6 = self.x6(x5, x4)
        x6_conv = self.x6_conv(x6)
        x7 = self.x7(x6_conv, x3)
        x7_conv = self.x7_conv(x7)
        x8 = self.x8(x7_conv, x2)
        x8_conv = self.x8_conv(x8)
        x9 = self.x9(x8_conv, x1)
        x9_conv = self.x9_conv(x9)

        # Final
        final = self.final(x9_conv)

        return final
