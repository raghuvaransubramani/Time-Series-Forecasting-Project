"""
model_lstm_attention.py
Defines AdditiveAttention and LSTMWithAttention PyTorch modules.
"""
import torch
import torch.nn as nn

class AdditiveAttention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim, attn_dim):
        super().__init__()
        self.W_enc = nn.Linear(enc_hidden_dim, attn_dim, bias=False)
        self.W_dec = nn.Linear(dec_hidden_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, enc_outputs, dec_hidden):
        # enc_outputs: (batch, seq_len, enc_hidden_dim)
        # dec_hidden: (batch, dec_hidden_dim)
        enc_proj = self.W_enc(enc_outputs)                     # (b, s, a)
        dec_proj = self.W_dec(dec_hidden).unsqueeze(1)         # (b, 1, a)
        score = self.v(torch.tanh(enc_proj + dec_proj)).squeeze(-1)  # (b, s)
        attn_weights = torch.softmax(score, dim=1)                # (b, s)
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs).squeeze(1) # (b, enc_hidden_dim)
        return context, attn_weights

class LSTMWithAttention(nn.Module):
    def __init__(self, input_dim, enc_hidden=128, dec_hidden=64, attn_dim=32, n_layers=1, out_dim=1):
        super().__init__()
        self.enc = nn.LSTM(input_dim, enc_hidden, n_layers, batch_first=True)
        self.attn = AdditiveAttention(enc_hidden, dec_hidden, attn_dim)
        self.dec_lstmcell = nn.LSTMCell(dec_hidden, dec_hidden)
        self.decoder_input_proj = nn.Linear(out_dim, dec_hidden)
        self.fc = nn.Linear(enc_hidden + dec_hidden, out_dim)
        self.dec_hidden_size = dec_hidden

    def forward(self, x, future_steps=1, teacher_forcing=None):
        enc_out, _ = self.enc(x)    # enc_out: (b, seq_len, enc_hidden)
        batch = x.size(0)
        dec_h = torch.zeros(batch, self.dec_hidden_size, device=x.device)
        dec_c = torch.zeros(batch, self.dec_hidden_size, device=x.device)
        prev_out = torch.zeros(batch, 1, device=x.device)
        preds = []
        attn_weights_all = []
        for t in range(future_steps):
            dec_in = self.decoder_input_proj(prev_out)  # (b, dec_hidden)
            dec_h, dec_c = self.dec_lstmcell(dec_in.squeeze(1), (dec_h, dec_c))
            context, attn_w = self.attn(enc_out, dec_h)
            combined = torch.cat([context, dec_h], dim=1)
            out = self.fc(combined)  # (b, out_dim)
            preds.append(out.unsqueeze(1))
            attn_weights_all.append(attn_w)
            if teacher_forcing is not None:
                prev_out = teacher_forcing[:, t:t+1]
            else:
                prev_out = out.detach()
        preds = torch.cat(preds, dim=1)  # (b, future_steps, out_dim)
        attn_stack = torch.stack(attn_weights_all, dim=1)  # (b, future_steps, seq_len)
        return preds, attn_stack
