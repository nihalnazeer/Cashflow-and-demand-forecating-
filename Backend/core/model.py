import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class HALSTM(nn.Module):
    def __init__(self, numerical_dim=12, hidden_dim=128, num_layers=2, num_heads=4, dropout=0.3, l2_lambda=0.01):
        super(HALSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.l2_lambda = l2_lambda
        self.input_dim = numerical_dim
        self.lstm = nn.LSTM(self.input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.lstm_norm = nn.LayerNorm(hidden_dim)
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.mha_norm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc_shared = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.positional_encoding = torch.zeros(3, hidden_dim, device=torch.device('cpu'))
        position = torch.arange(0, 3, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / hidden_dim))
        self.positional_encoding[:, 0::2] = torch.sin(position * div_term)
        self.positional_encoding[:, 1::2] = torch.cos(position * div_term)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name: nn.init.xavier_normal_(param)
                    elif 'bias' in name: nn.init.constant_(param, 0)

    def forward(self, numerical):
        batch_size, seq_len, _ = numerical.size()
        x = self.dropout(numerical)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        lstm_out = self.lstm_norm(lstm_out)
        lstm_out = lstm_out + self.positional_encoding[:seq_len, :].unsqueeze(0).to(x.device)
        mha_out, mha_weights = self.mha(lstm_out, lstm_out, lstm_out)
        mha_out = self.mha_norm(mha_out)
        combined = torch.cat([lstm_out[:, -1, :], mha_out[:, -1, :]], dim=-1)
        gate_val = self.sigmoid(self.gate(combined))
        fused = gate_val * lstm_out[:, -1, :] + (1 - gate_val) * mha_out[:, -1, :]
        shared = self.relu(self.fc_shared(fused))
        output = self.fc_out(shared)
        return output, {'mha_weights': mha_weights, 'gate_weights': gate_val}