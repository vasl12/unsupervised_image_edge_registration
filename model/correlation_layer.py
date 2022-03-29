
import torch
import torch.nn as nn

class Corr2D(nn.Module):
    def __init__(self, pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1, corr_multiply=1):
        assert kernel_size == 1
        assert pad_size == max_displacement
        assert stride1 == stride2 == 1
        super().__init__()
        self.max_dis = max_displacement
        self.pad = nn.ConstantPad2d(pad_size, 0)

    def forward(self, in1, in2):
        in2_pad = self.pad(in2)
        offsety, offsetx = torch.meshgrid([torch.arange(0, 2 * self.max_dis + 1),
                                           torch.arange(0, 2 * self.max_dis + 1)])
        hei, wid = in1.shape[2], in1.shape[3]
        output = torch.cat([
            torch.mean(in1 * in2_pad[:, :, dy:dy+hei, dx:dx+wid], 1, keepdim=True)  # here using mean instead of sum
            for dx, dy in zip(offsetx.reshape(-1), offsety.reshape(-1))
        ], 1)   # sliding window cross all pixels, mean of the feature dims, and then concatenate along the feature dim
        return output


# todo: need to be tested
class Corr3D(nn.Module):
    def __init__(self, pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1, stride3=1, corr_multiply=1):
        assert kernel_size == 1, "kernel_size other than 1 is not implemented"
        assert pad_size == max_displacement
        assert stride1 == stride2 == stride3 == 1
        super().__init__()
        self.max_dis = max_displacement
        self.pad = nn.ConstantPad3d(pad_size, 0)

    def forward(self, in1, in2):
        in2_pad = self.pad(in2)
        offsety, offsetx, offsetz = torch.meshgrid([torch.arange(0, 2 * self.max_dis + 1),
                                                    torch.arange(0, 2 * self.max_dis + 1),
                                                    torch.arange(0, 2 * self.max_dis + 1)])
        hei, wid, dep = in1.shape[2], in1.shape[3], in1.shape[4]  # todo: the order of x-, y-, z-axis need to be determined
        output = torch.cat([
            torch.sum(in1 * in2_pad[:, :, dy:dy+hei, dx:dx+wid, dz:dz+dep], 1, keepdim=True)
            for dx, dy, dz in zip(offsetx.reshape(-1), offsety.reshape(-1), offsetz.reshape(-1))
        ], 1)
        # sum = 0
        # for dx, dy, dz in zip(offsetx.reshape(-1), offsety.reshape(-1), offsetz.reshape(-1)):
        #     output = torch.cat([
        #         torch.mean(in1 * in2_pad[:, :, dy:dy + hei, dx:dx + wid, dz:dz + dep], 1, keepdim=True)], 1)
        #     sum = sum +1

        return output

