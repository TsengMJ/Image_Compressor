from torch import nn

class ConvUnit(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, padding):
        super(ConvUnit, self).__init__()
        self.conv1 = nn.Conv2d(input_size, output_size, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(output_size, output_size, kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(output_size, output_size, kernel_size=kernel_size, padding=padding)
        self.prelu  = nn.PReLU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.prelu(x)
        x = self.conv3(x)
        x = self.prelu(x)
        
        return x


class My_Model(nn.Module):
    def __init__(self):
        super(My_Model, self).__init__()
        self.convUnit1 = ConvUnit(input_size=3, output_size=128, kernel_size=3, padding=1)
        self.convUnit2 = ConvUnit(input_size=128, output_size=64, kernel_size=3, padding=1)
        self.convUnit3 = ConvUnit(input_size=64, output_size=32, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(32, 16, kernel_size=1)
        self.conv2 = nn.Conv2d(16, 3, kernel_size=5, padding=2)
        self.prelu  = nn.PReLU()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                print(12)
                nn.init.normal_(m.weight, std=0.001)

    def forward(self, x):
        x = self.convUnit1(x)
        x = self.convUnit2(x)
        x = self.convUnit3(x)
        x = self.conv1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        return x