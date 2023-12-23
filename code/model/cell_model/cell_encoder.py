import torch
import torch.nn as nn
import torch.nn.functional as F

class GE_vae(nn.Module):
    def __init__(self, model_config):
        super(GE_vae, self).__init__()
        # ENCODER fc layers
        # level 1
        # Expr input
        level_2_dim_expr = 4096
        level_3_dim_expr = 1024
        level_4_dim = 512
        self.hidden_dim = model_config.get('hidden_dim')
        self.e_fc1_expr = self.fc_layer(17419, level_2_dim_expr)
        # Level 2
        # self.e_fc2_expr = self.fc_layer(level_2_dim_expr, level_3_dim_expr)
        self.e_fc2_expr = self.fc_layer(level_2_dim_expr, level_3_dim_expr, dropout=True)

        # Level 3
        #self.e_fc3 = self.fc_layer(level_3_dim_expr, level_4_dim)
        self.e_fc3 = self.fc_layer(level_3_dim_expr, level_4_dim, dropout=True)

        # Level 4
        self.e_fc4_mean = self.fc_layer(level_4_dim, self.hidden_dim, activation=0)
        self.e_fc4_log_var = self.fc_layer(level_4_dim, self.hidden_dim, activation=0)

        # DECODER fc layers
        # Level 4
        self.d_fc4 = self.fc_layer(self.hidden_dim, level_4_dim)

        # Level 3
        # self.d_fc3 = self.fc_layer(level_4_dim, level_3_dim_expr)
        self.d_fc3 = self.fc_layer(level_4_dim, level_3_dim_expr, dropout=True)

        # Level 2
        # self.d_fc2_expr = self.fc_layer(level_3_dim_expr, level_2_dim_expr)
        self.d_fc2_expr = self.fc_layer(level_3_dim_expr, level_2_dim_expr, dropout=True)

        # level 1
        # Expr output
        self.d_fc1_expr = self.fc_layer(level_2_dim_expr, 8046, activation=2)
    # Activation - 0: no activation, 1: ReLU, 2: Sigmoid
    def fc_layer(self, in_dim, out_dim, activation=1, dropout=True, dropout_p=0.5):
        if activation == 0:
            layer = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim))
        elif activation == 2:
            layer = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.Sigmoid())
        else:
            if dropout:
                layer = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_p))
            else:
                layer = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU())
        return layer

    def encode(self, x):
        expr_level2_layer = self.e_fc1_expr(x)

        level_3_layer = self.e_fc2_expr(expr_level2_layer)

        level_4_layer = self.e_fc3(level_3_layer)

        latent_mean = self.e_fc4_mean(level_4_layer)
        latent_log_var = self.e_fc4_log_var(level_4_layer)

        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        sigma = torch.exp(0.5 * log_var)
        eps = torch.randn_like(sigma)
        return mean + eps * sigma

    def decode(self, z):
        level_4_layer = self.d_fc4(z)

        level_3_layer = self.d_fc3(level_4_layer)

        expr_level2_layer = self.d_fc2_expr(level_3_layer)

        recon_x = self.d_fc1_expr(expr_level2_layer)

        return recon_x
    
    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z)
        return z, recon_x, mean, log_var


class cell_cnn(nn.Module):
    def __init__(self, model_config, num_features) -> None:
        super().__init__()
        n_filters = 32 
        self.hidden_dim = model_config.get('hidden_dim')
        self.dropout_rate = model_config.get('dropout_rate')
        self.conv_xt_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8, bias = False)
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.pool_xt_1 = nn.MaxPool1d(3)
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters*2, kernel_size=8, bias = False)
        self.bn2 = nn.BatchNorm1d(n_filters*2)
        self.pool_xt_2 = nn.MaxPool1d(3)
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters*2, out_channels=n_filters*4, kernel_size=8, bias = False)
        self.bn3 = nn.BatchNorm1d(n_filters*4)
        self.pool_xt_3 = nn.MaxPool1d(3)
        self.out_dim = self.calculate_out_dim(num_features)
        self.fc1_xt = nn.Linear(self.out_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_rate)
    def calculate_out_dim(self, num_features, kernel_size=8, pool_size=3, n_filters=32, padding=0, stride=1, pool_stride=3):
        out_dim = num_features
        for _ in range(3):  # Iterating over 3 sets of Conv and Pool layers
            # Convolutional Layer Calculation
            out_dim = (out_dim - kernel_size + 2*padding) // stride + 1
            # Pooling Layer Calculation
            out_dim = out_dim // pool_size # Doubling number of filters for next Conv layer
        return out_dim * n_filters * 4 # Multiply by number of filters in the last Conv layer

    def forward(self,mut):
        conv_xt = self.bn1(self.conv_xt_1(mut.float()))
        conv_xt = F.relu(conv_xt)
        conv_xt = self.pool_xt_1(conv_xt)
        conv_xt = self.bn2(self.conv_xt_2(conv_xt))
        conv_xt = F.relu(conv_xt)
        conv_xt = self.pool_xt_2(conv_xt)
        conv_xt = self.bn3(self.conv_xt_3(conv_xt))
        conv_xt = F.relu(conv_xt)
        conv_xt = self.pool_xt_3(conv_xt)
        xt = conv_xt.view(-1, conv_xt.shape[1] * conv_xt.shape[2])
        xt = self.fc1_xt(xt)
        return xt