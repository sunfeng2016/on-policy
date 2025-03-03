import torch.nn as nn
from .util import init

"""CNN Modules and utils."""

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNNLayer_old(nn.Module):
    def __init__(self, obs_shape, hidden_size, use_orthogonal, use_ReLU, kernel_size=3, stride=1):
        super(CNNLayer_old, self).__init__()

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        input_channel = obs_shape[0]
        input_width = obs_shape[1]
        input_height = obs_shape[2]

        self.cnn = nn.Sequential(
            init_(nn.Conv2d(in_channels=input_channel,
                            out_channels=hidden_size // 2,
                            kernel_size=kernel_size,
                            stride=stride)
                  ),
            active_func,
            Flatten(),
            init_(nn.Linear(hidden_size // 2 * (input_width - kernel_size + stride) * (input_height - kernel_size + stride),
                            hidden_size)
                  ),
            active_func,
            init_(nn.Linear(hidden_size, hidden_size)), active_func)

    def forward(self, x):
        
        x = x / 255.0
        x = self.cnn(x)
        return x

class CNNLayer(nn.Module):
    def __init__(self, obs_shape, hidden_size, use_orthogonal, use_ReLU, kernel_size=3, stride=1):
        super(CNNLayer, self).__init__()

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        input_channel = obs_shape[0]
        input_width = obs_shape[1]
        input_height = obs_shape[2]

        # 卷积层
        self.cnn = nn.Sequential(
            init_(nn.Conv2d(in_channels=input_channel,
                            out_channels=hidden_size // 2,
                            kernel_size=kernel_size,
                            stride=stride)),
            active_func,
            # init_(nn.Conv2d(in_channels=hidden_size // 2,
            #                 out_channels=hidden_size,
            #                 kernel_size=kernel_size,
            #                 stride=stride)),
            # active_func,
            Flatten()
        )

        # 计算展平后的维度
        conv_output_width = (input_width - 2 * (kernel_size - 1)) // stride
        conv_output_height = (input_height - 2 * (kernel_size - 1)) // stride
        conv_output_dim = hidden_size * conv_output_width * conv_output_height
        
        conv_output_dim = hidden_size // 2 * (input_width - kernel_size + stride) * (input_height - kernel_size + stride)

        # 全连接层
        self.fc = nn.Sequential(
            init_(nn.Linear(conv_output_dim, hidden_size * 8)),  # (卷积后的维度, hidden_size * 8)
            active_func,
            init_(nn.Linear(hidden_size * 8, hidden_size * 4)),  # (hidden_size * 8, hidden_size * 4)
            active_func,
            init_(nn.Linear(hidden_size * 4, hidden_size)),      # (hidden_size * 4, hidden_size)
            active_func
        )

    def forward(self, x):
        # x = x / 255.0  # 将输入归一化
        x = x / 4.0  # 将输入归一化
        
        for l in self.cnn:
            x = l(x)
        
        # x = self.cnn(x)  # 卷积层
        x = self.fc(x)   # 全连接层
        return x


class CNNBase(nn.Module):
    def __init__(self, args, obs_shape):
        super(CNNBase, self).__init__()

        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self.hidden_size = args.hidden_size

        self.cnn = CNNLayer(obs_shape, self.hidden_size, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        x = self.cnn(x)
        return x
