import torch
import torch.nn as nn
import torch.nn.functional as fu


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, negative_slope=0.2):
        super(ConvBlock, self).__init__()
        assert ((kernel_size - stride) % 2 == 0)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=(kernel_size - stride) // 2
        )
        self.activation = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        return self.activation(self.conv(x))


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, negative_slope=0.2):
        super(DeconvBlock, self).__init__()
        assert ((kernel_size - stride) % 2 == 0)
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=(kernel_size - stride) // 2
        )
        self.activation = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        return self.activation(self.deconv(x))


class SEBlock(nn.Module):
    def __init__(self, channels):
        super(SEBlock, self).__init__()
        self.channels = channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.layer = nn.Sequential(
            nn.Linear(channels, channels), nn.ReLU(inplace=True),
            nn.Linear(channels, channels), nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pool(x).view(x.shape[0], self.channels)
        y = self.layer(y)
        y = y.view(x.shape[0], self.channels, 1, 1)
        return x * y


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, kernel_size):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(ConvBlock(in_channels + i * growth_rate, growth_rate, kernel_size))
        self.conv = nn.Conv2d(in_channels + growth_rate * num_layers, in_channels, kernel_size=1)
        self.se = SEBlock(in_channels)

    def forward(self, x):
        y = [x]
        for layer in self.layers:
            out = layer(torch.cat(y, dim=1))
            y.append(out)
        y = self.conv(torch.cat(y, dim=1))
        y = self.se(y)
        y = y + x
        return y


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.scale = config['scale']
        self.inner_channels = config['inner_channels']
        self.kernel_size_conv = config['kernel_size_conv']
        self.kernel_size_pool = config['kernel_size_pool']
        self.kernel_size_dense = config['kernel_size_dense']
        self.num_layers_dense = config['num_layers_dense']
        self.growth_rate_dense = config['growth_rate_dense']

        self.l1_conv = ConvBlock(4, self.inner_channels, self.kernel_size_conv)
        self.l2_dense = DenseBlock(
            self.inner_channels, self.growth_rate_dense, self.num_layers_dense, self.kernel_size_dense
        )
        self.l3_pool = ConvBlock(self.inner_channels, self.inner_channels, self.kernel_size_pool, stride=2)
        self.l4_dense = DenseBlock(
            self.inner_channels, self.growth_rate_dense, self.num_layers_dense, self.kernel_size_dense
        )
        self.l5_dense = DenseBlock(
            self.inner_channels, self.growth_rate_dense, self.num_layers_dense, self.kernel_size_dense
        )
        self.l6_bottle = ConvBlock(self.inner_channels * 3, self.inner_channels, 1)

        self.l7_deconv = DeconvBlock(self.inner_channels, self.inner_channels, self.kernel_size_pool, stride=2)
        self.l8_conv = ConvBlock(self.inner_channels, self.inner_channels, self.kernel_size_conv)
        self.l9_dense = DenseBlock(
            self.inner_channels, self.growth_rate_dense, self.num_layers_dense, self.kernel_size_dense
        )
        self.l10_bottle = ConvBlock(self.inner_channels * 4, self.inner_channels, self.kernel_size_conv)

        self.l11_conv = ConvBlock(self.inner_channels, 16, self.kernel_size_conv)

    def forward(self, x0):
        x1 = self.l1_conv(x0)
        x2 = self.l2_dense(x1)
        x3 = self.l3_pool(x2)
        x4 = self.l4_dense(x3)
        x5 = self.l5_dense(x4)
        x6 = self.l6_bottle(torch.cat([x3, x4, x5], dim=1))

        x7 = self.l7_deconv(x6)
        x8 = self.l8_conv(x7)
        x9 = self.l9_dense(x8)
        x10 = self.l10_bottle(torch.cat([x1, x2, x8, x9], dim=1))

        x11 = self.l11_conv(x10)

        c1 = fu.pixel_shuffle(x11[:, :4], 2)
        c2 = fu.pixel_shuffle(x11[:, 4:8], 2)
        c3 = fu.pixel_shuffle(x11[:, 8:12], 2)
        c4 = fu.pixel_shuffle(x11[:, 12:], 2)
        x12 = torch.cat([c1, c2, c3, c4], dim=1)
        return x12


class Branch2(nn.Module):
    def __init__(self, config):
        super(Branch2, self).__init__()
        self.scale = config['scale']
        self.inner_channels = config['inner_channels']
        self.kernel_size_conv = config['kernel_size_conv']
        self.kernel_size_pool = config['kernel_size_pool']
        self.kernel_size_dense = config['kernel_size_dense']
        self.num_layers_dense = config['num_layers_dense']
        self.growth_rate_dense = config['growth_rate_dense']

        self.l1_conv = ConvBlock(4, self.inner_channels, self.kernel_size_conv)

        self.l2_pool = ConvBlock(self.inner_channels, self.inner_channels, self.kernel_size_pool)
        self.l3_conv = ConvBlock(self.inner_channels, self.inner_channels, self.kernel_size_conv)

        self.l4_pool = ConvBlock(self.inner_channels, self.inner_channels, self.kernel_size_pool)
        self.l5_conv = ConvBlock(self.inner_channels, self.inner_channels, self.kernel_size_conv)

        self.l6_deconv = DeconvBlock(self.inner_channels, self.inner_channels, self.kernel_size_conv)
        self.l7_conv = De


    def forward(self, x0):
        x1 = self.l1_conv(x0)
        x2 = self.l2_conv(x1)
        x3 = self.l3_conv(x2)
        x4 = self.l4_conv(x3)
        x5 = self.l5_conv(x4)



    # def branch_2_0(self):
    #     with tf.variable_scope('branch_2_0', reuse=self.reuse):
    #         conv_2_0_0 = slim.conv2d(inputs=self.input_isp, num_outputs=128, kernel_size=self.kernel_size,
    #                                  reuse=self.reuse, scope='conv_2_0_0', activation_fn=model_tools.lrelu)
    #         pool_2_0_0 = slim.avg_pool2d(inputs=conv_2_0_0, kernel_size=2, scope='pool_2_0_0')
    #
    #         conv_2_0_1 = slim.conv2d(inputs=pool_2_0_0, num_outputs=128, kernel_size=self.kernel_size,
    #                                  reuse=self.reuse, scope='conv_2_0_1', activation_fn=model_tools.lrelu)
    #         pool_2_0_1 = slim.avg_pool2d(inputs=conv_2_0_1, kernel_size=2, scope='pool_2_0_1')
    #
    #         conv_2_0_2 = slim.conv2d(inputs=pool_2_0_1, num_outputs=128, kernel_size=self.kernel_size,
    #                                  reuse=self.reuse, scope='conv_2_0_2', activation_fn=model_tools.lrelu)
    #         fc21 = tf.concat([conv_2_0_2, pool_2_0_1], axis=3)
    #         return fc21, pool_2_0_0
    #
    # def branch_2_1(self, fcin21, pool_2_0_0):
    #     with tf.variable_scope('branch_2_1', reuse=self.reuse):
    #         conv_2_1_2_nFF = slim.conv2d(inputs=fcin21, num_outputs=256, kernel_size=3, reuse=self.reuse, stride=1,
    #                                      scope='conv_2_1_2_nFF', activation_fn=model_tools.lrelu)
    #         deconv_2_1_0 = slim.conv2d_transpose(inputs=conv_2_1_2_nFF, num_outputs=128,
    #                                              kernel_size=[4, 4], reuse=self.reuse, stride=2, scope='deconv_2_0_0',
    #                                              activation_fn=model_tools.lrelu)
    #         conv_2_1_3 = slim.conv2d(inputs=deconv_2_1_0, num_outputs=128, kernel_size=self.kernel_size,
    #                                  reuse=self.reuse, scope='conv_2_0_3', activation_fn=model_tools.lrelu)
    #         fc22 = tf.concat([conv_2_1_3, pool_2_0_0], axis=3)
    #         return fc22
    #
    # def branch_2_2(self, fcin22):
    #     with tf.variable_scope('branch_2_2', reuse=self.reuse):
    #         conv_2_1_3_nFF = slim.conv2d(inputs=fcin22, num_outputs=256, kernel_size=3, reuse=self.reuse, stride=1,
    #                                      scope='conv_2_1_3_nFF', activation_fn=model_tools.lrelu)
    #         deconv_2_1_1 = slim.conv2d_transpose(inputs=conv_2_1_3_nFF, num_outputs=128,
    #                                              kernel_size=[4, 4], reuse=self.reuse, stride=2, scope='deconv_2_0_1',
    #                                              activation_fn=model_tools.lrelu)
    #         conv_2_1_4 = slim.conv2d(inputs=deconv_2_1_1, num_outputs=128, kernel_size=self.kernel_size,
    #                                  reuse=self.reuse, scope='deconv_2_0_4', activation_fn=model_tools.lrelu)
    #
    #         deconv_2_1_2 = slim.conv2d_transpose(inputs=conv_2_1_4, num_outputs=128, kernel_size=[4, 4],
    #                                              reuse=self.reuse,
    #                                              stride=Scale, scope='deconv_2_0_2', activation_fn=model_tools.lrelu)
    #         conv_2_1_5 = slim.conv2d(inputs=deconv_2_1_2, num_outputs=12, kernel_size=self.kernel_size,
    #                                  reuse=self.reuse,
    #                                  scope='conv_2_0_5', activation_fn=None)
    #
    #         color_matrix_per_pixel = conv_2_1_5
    #         return color_matrix_per_pixel


if __name__ == '__main__':
    pass
