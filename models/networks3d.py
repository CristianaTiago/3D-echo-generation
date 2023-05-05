import torch
import torch.nn as nn
import torch.nn.functional
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm3d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    #if which_model_netG == 'resnet_9blocks':
    #     netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    # elif which_model_netG == 'resnet_6blocks':
    #     netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    # elif which_model_netG == 'unet_128':
    #     netG = UnetGenerator3d(input_nc, output_nc, 5, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    if which_model_netG == 'unet_256':
        netG = UNet(input_nc, output_nc, 5)
    #elif which_model_netG == 'vnet':
    #    netG = VNet(elu=False, nll=True)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(device=gpu_ids[0])
    netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda(device=gpu_ids[0])
    netD.apply(weights_init)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.use_lsgan = use_lsgan
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()
        print(self.loss)

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

    
# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
# class UnetGenerator(nn.Module):
#     def __init__(self, input_nc, output_nc, num_downs, ngf=64,
#                  norm_layer=nn.BatchNorm3d, use_dropout=False, gpu_ids=[]):
#         super(UnetGenerator, self).__init__()
#         self.gpu_ids = gpu_ids
#
#         # currently support only input_nc == output_nc
#         assert(input_nc == output_nc)
#
#         # construct unet structure
#         unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)
#         for i in range(num_downs - 5):
#             unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
#         unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
#         unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
#         unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
#         unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)
#
#         self.model = unet_block
#
#     def forward(self, input):
#         if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
#             return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
#         else:
#             return self.model(input)

#############################
#
# 3D version of UnetGenerator
# class UnetGenerator3d(nn.Module):
#     def __init__(self, input_nc, output_nc, num_downs, ngf=64,
#                  norm_layer=nn.BatchNorm3d, use_dropout=False, gpu_ids=[]): # TODO
#         super(UnetGenerator3d, self).__init__()
#         self.gpu_ids = gpu_ids
#
#         # currently support only input_nc == output_nc
#         assert(input_nc == output_nc)
#
#         # construct unet structure
#         unet_block = UnetSkipConnectionBlock3d(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)
#         for i in range(num_downs - 5):
#             unet_block = UnetSkipConnectionBlock3d(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
#         unet_block = UnetSkipConnectionBlock3d(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
#         unet_block = UnetSkipConnectionBlock3d(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
#         unet_block = UnetSkipConnectionBlock3d(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
#         unet_block = UnetSkipConnectionBlock3d(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)
#
#         self.model = unet_block
#
#     def forward(self, input):
#         if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
#             return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
#         else:
#             return self.model(input)
#
# # Defines the submodule with skip connection.
# # X -------------------identity---------------------- X
# #   |-- downsampling -- |submodule| -- upsampling --|
# class UnetSkipConnectionBlock3d(nn.Module):
#     def __init__(self, outer_nc, inner_nc,
#                  submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm3d, use_dropout=False):
#         super(UnetSkipConnectionBlock3d, self).__init__()
#         self.outermost = outermost
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm3d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm3d
#
#         downconv = nn.Conv3d(outer_nc, inner_nc, kernel_size=2,
#                              stride=2, padding=1, bias=use_bias)
#         downrelu = nn.LeakyReLU(0.2, True)
#         downnorm = norm_layer(inner_nc)
#         uprelu = nn.ReLU(True)
#         upnorm = norm_layer(outer_nc)
#
#         if outermost:
#             upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
#                                         kernel_size=2, stride=2,
#                                         padding=1, output_padding=1)
#             down = [downconv]
#             up = [uprelu, upconv, nn.Tanh()]
#             model = down + [submodule] + up
#         elif innermost:
#             upconv = nn.ConvTranspose3d(inner_nc, outer_nc,
#                                         kernel_size=2, stride=2,
#                                         padding=1, bias=use_bias, output_padding=1)
#             down = [downrelu, downconv]
#             up = [uprelu, upconv, upnorm]
#             model = down + up
#         else:
#             upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
#                                         kernel_size=2, stride=2,
#                                         padding=1, bias=use_bias)
#             down = [downrelu, downconv, downnorm]
#             up = [uprelu, upconv, upnorm]
#
#             if use_dropout:
#                 model = down + [submodule] + up + [nn.Dropout(0.5)]
#             else:
#                 model = down + [submodule] + up
#
#         self.model = nn.Sequential(*model)
#
#     def forward(self, x):
#         print(self.model(x).shape, x.shape)
#         if self.outermost:
#             return self.model(x)
#         else:
#             print(self.model.modules())
#             return torch.cat([self.model(x), x], 1)
#
# # Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 2
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)




def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        activation, )


def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        activation, )

        #nn.Upsample(scale_factor = 2, mode='trilinear', align_corners=True)



        #nn.Sequential(
        #nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
        #nn.Conv3d(in_dim, out_dim*2, 3, stride=1, padding=1),
        #nn.BatchNorm3d(out_dim*2),
        #nn.ReLU(inplace=True))




def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


def conv_block_2_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        activation)


class UNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):
        super(UNet, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.LeakyReLU(0.2, inplace=True)

        # Down sampling
        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filters, activation)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d(self.num_filters, self.num_filters * 2, activation)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation)
        self.pool_3 = max_pooling_3d()
        self.down_4 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation)
        self.pool_4 = max_pooling_3d()
        self.down_5 = conv_block_2_3d(self.num_filters * 8, self.num_filters * 16, activation)
        self.pool_5 = max_pooling_3d()

        # Bridge
        self.bridge = conv_block_2_3d(self.num_filters * 16, self.num_filters * 32, activation)

        # Up sampling
        self.trans_1 = conv_trans_block_3d(self.num_filters * 32, self.num_filters * 32, activation)
        self.up_1 = conv_block_2_3d(self.num_filters * 48, self.num_filters * 16, activation)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 16, self.num_filters * 16, activation)
        self.up_2 = conv_block_2_3d(self.num_filters * 24, self.num_filters * 8, activation)
        self.trans_3 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation)
        self.up_3 = conv_block_2_3d(self.num_filters * 12, self.num_filters * 4, activation)
        self.trans_4 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation)
        self.up_4 = conv_block_2_3d(self.num_filters * 6, self.num_filters * 2, activation)
        self.trans_5 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation)
        self.up_5 = conv_block_2_3d(self.num_filters * 3, self.num_filters * 1, activation)

        # Output
        self.out = conv_block_3d(self.num_filters, out_dim, activation)

    def forward(self, x):
        # Down sampling
        down_1 = self.down_1(x)  # -> [1, 4, 128, 128, 128]
        #print('down_1 =', down_1.shape)
        pool_1 = self.pool_1(down_1)  # -> [1, 4, 64, 64, 64]
        #print('pool_1 =', pool_1.shape)
        down_2 = self.down_2(pool_1)  # -> [1, 8, 64, 64, 64]
        #print('down_2 =', down_2.shape)
        pool_2 = self.pool_2(down_2)  # -> [1, 8, 32, 32, 32]
        #print('pool_2 =', pool_2.shape)
        down_3 = self.down_3(pool_2)  # -> [1, 16, 32, 32, 32]
        #print('down_3 =', down_3.shape)
        pool_3 = self.pool_3(down_3)  # -> [1, 16, 16, 16, 16]
        #print('pool_3 =', pool_3.shape)
        down_4 = self.down_4(pool_3)  # -> [1, 32, 16, 16, 16]
        #print('down_4 =', down_4.shape)
        pool_4 = self.pool_4(down_4)  # -> [1, 32, 8, 8, 8]
        #print('pool_4 =', pool_4.shape)
        down_5 = self.down_5(pool_4)  # -> [1, 64, 8, 8, 8]
        #print('down_5 =', down_5.shape)
        pool_5 = self.pool_5(down_5)  # -> [1, 64, 4, 4, 4]
        #print('pool_5 =', pool_5.shape)
        # Bridge
        bridge = self.bridge(pool_5)  # -> [1, 128, 4, 4, 4]
        #print('bridge =', bridge.shape)
        # Up sampling
        trans_1 = self.trans_1(bridge)
        #print('trans_1 =', trans_1.shape)# -> [1, 128, 8, 8, 8]
        concat_1 = torch.cat([trans_1, down_5], dim=1)  # -> [1, 192, 8, 8, 8]
        #print('concat_1 =', concat_1.shape)
        up_1 = self.up_1(concat_1)  # -> [1, 64, 8, 8, 8]
        #print('up_1 =', up_1.shape)
        trans_2 = self.trans_2(up_1)  # -> [1, 64, 16, 16, 16]
        #print(trans_2.shape, down_4.shape)
        concat_2 = torch.cat([trans_2, down_4], dim=1)  # -> [1, 96, 16, 16, 16]
        up_2 = self.up_2(concat_2)  # -> [1, 32, 16, 16, 16]

        trans_3 = self.trans_3(up_2)  # -> [1, 32, 32, 32, 32]
        concat_3 = torch.cat([trans_3, down_3], dim=1)  # -> [1, 48, 32, 32, 32]
        up_3 = self.up_3(concat_3)  # -> [1, 16, 32, 32, 32]

        trans_4 = self.trans_4(up_3)  # -> [1, 16, 64, 64, 64]
        concat_4 = torch.cat([trans_4, down_2], dim=1)  # -> [1, 24, 64, 64, 64]
        up_4 = self.up_4(concat_4)  # -> [1, 8, 64, 64, 64]

        trans_5 = self.trans_5(up_4)  # -> [1, 8, 128, 128, 128]
        concat_5 = torch.cat([trans_5, down_1], dim=1)  # -> [1, 12, 128, 128, 128]
        up_5 = self.up_5(concat_5)  # -> [1, 4, 128, 128, 128]

        # Output
        out = self.out(up_5)  # -> [1, 3, 128, 128, 128]
        return out
