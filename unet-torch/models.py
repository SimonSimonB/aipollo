import torch

class SimpleConv(torch.nn.Module):
    def __init__(self):
        super(SimpleConv, self).__init__()

        self._conv1 = torch.nn.Conv2d(1, 1, 25, padding=12)
    
    def forward(self, inputs):
        return torch.nn.functional.sigmoid(self._conv1(inputs))


class DeepConv(torch.nn.Module):
    def __init__(self, num_channels=32):
        super(DeepConv, self).__init__()

        self._conv1 = torch.nn.Conv2d(1, num_channels, 3, padding=1)
        self._conv2 = torch.nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self._conv3 = torch.nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self._conv4 = torch.nn.Conv2d(num_channels, 1, 1)
    
    def forward(self, inputs):
        return torch.nn.functional.sigmoid(self._conv4(self._conv3(self._conv2(self._conv1(inputs)))))


class UNet(torch.nn.Module):
    def __init__(self):
        FIRST_LAYER_CHANNELS = 8

        super(UNet, self).__init__()
        self.cd1a = torch.nn.Conv2d(1, FIRST_LAYER_CHANNELS, 3, padding=1)
        self.cd1b = torch.nn.Conv2d(FIRST_LAYER_CHANNELS, FIRST_LAYER_CHANNELS, 3, padding=1)
        
        self.cd2a = torch.nn.Conv2d(FIRST_LAYER_CHANNELS, 2 * FIRST_LAYER_CHANNELS, 3, padding=1)
        self.cd2b = torch.nn.Conv2d(2 * FIRST_LAYER_CHANNELS, 2 * FIRST_LAYER_CHANNELS, 3, padding=1)
        
        self.cd3a = torch.nn.Conv2d(2 * FIRST_LAYER_CHANNELS, 4 * FIRST_LAYER_CHANNELS, 3, padding=1)
        self.cd3b = torch.nn.Conv2d(4 * FIRST_LAYER_CHANNELS, 4 * FIRST_LAYER_CHANNELS, 3, padding=1)

        self.cd4a = torch.nn.Conv2d(4 * FIRST_LAYER_CHANNELS, 8 * FIRST_LAYER_CHANNELS, 3, padding=1)
        self.cd4b = torch.nn.Conv2d(8 * FIRST_LAYER_CHANNELS, 8 * FIRST_LAYER_CHANNELS, 3, padding=1)

        self.cma = torch.nn.Conv2d(8 * FIRST_LAYER_CHANNELS, 16 * FIRST_LAYER_CHANNELS, 3, padding=1)
        self.cmb = torch.nn.Conv2d(16 * FIRST_LAYER_CHANNELS, 16 * FIRST_LAYER_CHANNELS, 3, padding=1)
        
        self.deconvto4 = torch.nn.ConvTranspose2d(16 * FIRST_LAYER_CHANNELS, 8 * FIRST_LAYER_CHANNELS, 2, stride=2)
        self.cu4a = torch.nn.Conv2d(16 * FIRST_LAYER_CHANNELS, 8 * FIRST_LAYER_CHANNELS, 3, padding=1)
        self.cu4b = torch.nn.Conv2d(8 * FIRST_LAYER_CHANNELS, 8 * FIRST_LAYER_CHANNELS, 3, padding=1)

        self.deconvto3 = torch.nn.ConvTranspose2d(8 * FIRST_LAYER_CHANNELS, 4 * FIRST_LAYER_CHANNELS, 2, stride=2)
        self.cu3a = torch.nn.Conv2d(8 * FIRST_LAYER_CHANNELS, 4 * FIRST_LAYER_CHANNELS, 3, padding=1)
        self.cu3b = torch.nn.Conv2d(4 * FIRST_LAYER_CHANNELS, 4 * FIRST_LAYER_CHANNELS, 3, padding=1)

        self.deconvto2 = torch.nn.ConvTranspose2d(4 * FIRST_LAYER_CHANNELS, 2 * FIRST_LAYER_CHANNELS, 2, stride=2)
        self.cu2a = torch.nn.Conv2d(4 * FIRST_LAYER_CHANNELS, 2 * FIRST_LAYER_CHANNELS, 3, padding=1)
        self.cu2b = torch.nn.Conv2d(2 * FIRST_LAYER_CHANNELS, 2 * FIRST_LAYER_CHANNELS, 3, padding=1)
        
        self.deconvto1 = torch.nn.ConvTranspose2d(2 * FIRST_LAYER_CHANNELS, FIRST_LAYER_CHANNELS, 2, stride=2)
        self.cu1a = torch.nn.Conv2d(2 * FIRST_LAYER_CHANNELS, FIRST_LAYER_CHANNELS, 3, padding=1)
        self.cu1b = torch.nn.Conv2d(FIRST_LAYER_CHANNELS, FIRST_LAYER_CHANNELS, 3, padding=1)

        self.output_conv = torch.nn.Conv2d(FIRST_LAYER_CHANNELS, 1, 1)
        
    def forward(self, inputs):
        d1_result = torch.nn.functional.relu(self.cd1b(self.cd1a(inputs)))
        to_d2 = torch.nn.functional.max_pool2d(d1_result, 2)
        d2_result = torch.nn.functional.relu(self.cd2b(self.cd2a(to_d2)))
        to_d3 = torch.nn.functional.max_pool2d(d2_result, 2)
        d3_result = torch.nn.functional.relu(self.cd3b(self.cd3a(to_d3)))
        to_d4 = torch.nn.functional.max_pool2d(d3_result, 2)
        d4_result = torch.nn.functional.relu(self.cd4b(self.cd4a(to_d4)))
        to_m = torch.nn.functional.max_pool2d(d4_result, 2)

        m_result = torch.nn.functional.relu(self.cmb(self.cma(to_m)))
        to_u4 = self.deconvto4(m_result)

        concatenated_u4 = torch.cat((d4_result, to_u4), dim=1)
        u4_result = torch.nn.functional.relu(self.cu4b(self.cu4a(concatenated_u4)))
        to_u3 = self.deconvto3(u4_result)

        concatenated_u3 = torch.cat((d3_result, to_u3), dim=1)
        u3_result = torch.nn.functional.relu(self.cu3b(self.cu3a(concatenated_u3)))
        to_u2 = self.deconvto2(u3_result)

        concatenated_u2 = torch.cat((d2_result, to_u2), dim=1)
        u2_result = torch.nn.functional.relu(self.cu2b(self.cu2a(concatenated_u2)))
        to_u1 = self.deconvto1(u2_result)

        concatenated_u1 = torch.cat((d1_result, to_u1), dim=1)
        u1_result = torch.nn.functional.relu(self.cu1b(self.cu1a(concatenated_u1)))
        output_conv = torch.sigmoid(self.output_conv(u1_result))

        return output_conv
