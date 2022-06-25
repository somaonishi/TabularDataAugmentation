import torch
import torch.nn as nn
import torch.nn.functional as F


class TCN(nn.Module):
    def __init__(self, input_dim, out_dim, sign_size=64, cha_input=16, cha_hidden=32,
                 K=2, dropout_input=0.2, dropout_hidden=0.2):
        super().__init__()

        hidden_size = sign_size * cha_input
        sign_size1 = sign_size
        sign_size2 = sign_size // 2
        out_features = (sign_size // 4) * cha_hidden

        self.hidden_size = hidden_size
        self.cha_input = cha_input
        self.cha_hidden = cha_hidden
        self.K = K
        self.sign_size1 = sign_size1
        self.sign_size2 = sign_size2
        self.out_features = out_features
        self.dropout_input = dropout_input
        self.dropout_hidden = dropout_hidden

        self.batch_norm0 = nn.BatchNorm1d(input_dim)
        self.dropout0 = nn.Dropout(dropout_input)
        dense0 = nn.Linear(input_dim, hidden_size, bias=False)
        self.dense0 = nn.utils.weight_norm(dense0)

        self.batch_norm1 = nn.BatchNorm1d(hidden_size)

        # 1st conv layer
        self.batch_norm_c1 = nn.BatchNorm1d(cha_input)
        conv1 = conv1 = nn.Conv1d(
            cha_input,
            cha_input*K,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=cha_input,
            bias=False)
        self.conv1 = nn.utils.weight_norm(conv1, dim=None)

        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size=sign_size2)

        # 2nd conv layer
        self.batch_norm_c2 = nn.BatchNorm1d(cha_input*K)
        self.dropout_c2 = nn.Dropout(dropout_hidden)
        conv2 = nn.Conv1d(
            cha_input*K,
            cha_hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.conv2 = nn.utils.weight_norm(conv2, dim=None)

        # 3rd conv layer
        self.batch_norm_c3 = nn.BatchNorm1d(cha_hidden)
        self.dropout_c3 = nn.Dropout(dropout_hidden)
        conv3 = nn.Conv1d(
            cha_hidden,
            cha_hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.conv3 = nn.utils.weight_norm(conv3, dim=None)

        # 4th conv layer
        self.batch_norm_c4 = nn.BatchNorm1d(cha_hidden)
        conv4 = nn.Conv1d(
            cha_hidden,
            cha_hidden,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=cha_hidden,
            bias=False)
        self.conv4 = nn.utils.weight_norm(conv4, dim=None)

        self.avg_po_c4 = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()

        self.classifier = nn.Linear(out_features, out_dim)

    def forward(self, x: torch.Tensor, transform=None):
        # x = self.batch_norm0(x)
        # x = self.dropout0(x)
        x = F.celu(self.dense0(x))

        if transform is not None:
            x = transform(x)

        x = self.batch_norm1(x)

        x = x.reshape(x.shape[0], self.cha_input, self.sign_size1)

        x = self.batch_norm_c1(x)
        x = F.relu(self.conv1(x))

        x = self.ave_po_c1(x)

        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = F.relu(self.conv2(x))
        x_s = x

        x = self.batch_norm_c3(x)
        x = self.dropout_c3(x)
        x = F.relu(self.conv3(x))

        x = self.batch_norm_c4(x)
        x = self.conv4(x)
        x = x + x_s
        x = F.relu(x)

        x = self.avg_po_c4(x)

        x = self.flt(x)

        x = self.classifier(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, h_dim, bn_before_augment=False, use_bn=True) -> None:
        super().__init__()
        self.l1 = nn.Linear(in_dim, h_dim)
        self.bn_before_augment = None
        if bn_before_augment:
            self.bn_before_augment = nn.BatchNorm1d(h_dim)
        self.l2 = nn.Linear(h_dim, h_dim)
        self.bn = None
        if use_bn:
            self.bn = nn.BatchNorm1d(h_dim)
        self.l3 = nn.Linear(h_dim, out_dim)

    def forward(self, x: torch.Tensor, transform=None):
        x = F.relu(self.l1(x))
        if self.bn_before_augment is not None:
            x = self.bn_before_augment(x)

        if transform is not None:
            x = transform(x)

        if self.bn is not None:
            x = self.bn(x)
        x = F.relu(self.l2(x))
        return self.l3(x)


def get_model(name, in_dim, out_dim, h_dim=128, bn_before_augment=False, use_bn=True):
    if name == "TCN":
        return TCN(in_dim, out_dim)
    if name == "MLP":
        return MLP(in_dim, out_dim, h_dim, bn_before_augment, use_bn)


if __name__ == '__main__':
    bn = nn.BatchNorm1d(10)
    print(bn._parameters)
