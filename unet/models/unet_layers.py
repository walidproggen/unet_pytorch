import torch


class ConvBlock(torch.nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, padding='same', stride=1):
        super().__init__()

        # Building the conv-block
        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding, bias=False),
            torch.nn.BatchNorm2d(output_dim),
            torch.nn.ReLU(),
            torch.nn.Conv2d(output_dim, output_dim, kernel_size, stride, padding, bias=False),
            torch.nn.BatchNorm2d(output_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.0)
        )

    def forward(self, x):
        return self.conv_block(x)


class UpConvBlock(torch.nn.Module):
    def __init__(self, input_dim, padding='same'):
        super().__init__()
        output_dim = int(input_dim / 2.)

        # Building the conv-block
        self.up_conv_block = torch.nn.Sequential(
            torch.nn.UpsamplingNearest2d(scale_factor=2),
            torch.nn.Conv2d(input_dim, output_dim, kernel_size=2, padding=padding, bias=False),
            torch.nn.BatchNorm2d(output_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.0)
        )

    def forward(self, x1, x2):
        x = self.up_conv_block(x1)
        concatenated = torch.cat((x2, x), dim=1)
        return concatenated
