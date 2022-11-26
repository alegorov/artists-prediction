import numpy as np
import pandas as pd
import os
from argparse import ArgumentParser
import torch
import torch.nn as nn
from tqdm import tqdm
import random
import torch.nn.functional as F
import torchaudio


class FeaturesLoader:
    def __init__(self, features_dir_path, meta_info, device='cpu'):
        self.features_dir_path = features_dir_path
        self.meta_info = meta_info
        self.trackid2path = meta_info.set_index('trackid')['archive_features_path'].to_dict()
        self.device = device

    def _load_item(self, track_id):
        track_features_file_path = self.trackid2path[track_id]
        track_features = np.load(os.path.join(self.features_dir_path, track_features_file_path))
        track_features = np.transpose(track_features)
        N = 3
        track_features = np.concatenate([track_features] * N, axis=0)[:61 * N]
        return track_features

    def load_batch(self, tracks_ids):
        batch = [self._load_item(track_id) for track_id in tracks_ids]
        return torch.tensor(np.array(batch)).to(self.device)


class TestLoader:
    def __init__(self, features_loader, batch_size=256):
        self.features_loader = features_loader
        self.batch_size = batch_size

    def __iter__(self):
        batch_ids = []
        for track_id in tqdm(self.features_loader.meta_info['trackid'].values):
            batch_ids.append(track_id)
            if len(batch_ids) == self.batch_size:
                yield batch_ids, self.features_loader.load_batch(batch_ids)
                batch_ids = []
        if len(batch_ids) > 0:
            yield batch_ids, self.features_loader.load_batch(batch_ids)


class BasicNet1(nn.Module):
    def __init__(self, output_features_size):
        super().__init__()
        self.output_features_size = output_features_size

        self.rnn1 = nn.LSTM(512, output_features_size, num_layers=1)
        self.rnn2 = nn.GRU(512, output_features_size, num_layers=1)
        # self.linear1 = nn.Linear(output_features_size, output_features_size, bias=False)
        # self.linear2 = nn.Linear(output_features_size, output_features_size, bias=False)
        # self.linear3 = nn.Linear(output_features_size, output_features_size, bias=False)
        # self.linear4 = nn.Linear(output_features_size, output_features_size, bias=False)

    def forward(self, x):
        x = torch.transpose(x, 0, 1)

        x1, _ = self.rnn1(x)
        x2, _ = self.rnn2(x)

        x = x1.mean(axis=0) + x2.mean(axis=0)

        # x = self.linear1(x)
        # x = self.linear2(torch.tanh(x))
        # x = self.linear3(torch.tanh(x))
        # x = self.linear4(torch.tanh(x))

        return x


class BasicNet2(nn.Module):
    def __init__(self, output_features_size):
        super().__init__()
        self.output_features_size = output_features_size

        nhead = 8
        dim_feedforward = 2048

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(512, nhead=nhead, dim_feedforward=dim_feedforward, dropout=0.25,
                                       batch_first=True), num_layers=3)
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x = torch.transpose(x, 1, 2)
        # x = x.permute((0, 2, 1))
        # x = self.transformer(x.float())

        x = self.transformer(x)
        x = x.permute((0, 2, 1))
        x = self.pooling(x)
        x = x.squeeze()
        return x


class BasicNet3(nn.Module):
    def __init__(self, output_features_size):
        super().__init__()
        self.output_features_size = output_features_size

        self.conv_1 = nn.Conv1d(512, 2 * output_features_size, kernel_size=4)
        self.conv_2 = nn.Conv1d(2 * output_features_size, output_features_size, kernel_size=3)

        self.conv_3 = nn.Conv1d(output_features_size, 2 * output_features_size, kernel_size=4)
        self.conv_4 = nn.Conv1d(2 * output_features_size, output_features_size, kernel_size=3)

        self.conv_5 = nn.Conv1d(output_features_size, 2 * output_features_size, kernel_size=3)
        self.conv_6 = nn.Conv1d(2 * output_features_size, output_features_size, kernel_size=3)

        self.conv_7 = nn.Conv1d(output_features_size, output_features_size, kernel_size=3)

        self.mp = nn.MaxPool1d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.transpose(x, 1, 2)

        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        m1 = x.mean(axis=2)
        x = self.relu(x)
        x = self.mp(x)

        x = self.conv_3(x)
        x = self.relu(x)
        x = self.conv_4(x)
        m2 = x.mean(axis=2)
        x = self.relu(x)
        x = self.mp(x)

        x = self.conv_5(x)
        x = self.relu(x)
        x = self.conv_6(x)
        m3 = x.mean(axis=2)
        x = self.relu(x)
        x = self.mp(x)

        x = self.conv_7(x)
        m4 = x.mean(axis=2)

        x = m1 + m2 + m3 + m4

        return x


class BasicNet4(nn.Module):
    def __init__(self, output_features_size):
        super().__init__()
        self.output_features_size = output_features_size

        self.emformer = torchaudio.models.Emformer(
            input_dim=512,
            num_heads=4,
            ffn_dim=128,
            num_layers=4,
            segment_length=4,
            right_context_length=1
        )

    def forward(self, x):
        lengths = torch.tensor([x.shape[1]] * x.shape[0], dtype=torch.int64, device=x.device)

        x = self.emformer(x, lengths)
        x = x[0].mean(axis=1)

        return x


class BasicNet5(nn.Module):
    def __init__(self, output_features_size):
        super().__init__()
        self.output_features_size = output_features_size

        self.conformer = torchaudio.models.Conformer(
            input_dim=512,
            num_heads=4,
            ffn_dim=128,
            num_layers=4,
            depthwise_conv_kernel_size=31,
        )

    def forward(self, x):
        lengths = torch.tensor([x.shape[1]] * x.shape[0], dtype=torch.int64, device=x.device)

        x = self.conformer(x, lengths)
        x = x[0].mean(axis=1)

        return x


def inference(model, loader):
    embeds = dict()
    model.eval()
    for tracks_ids, tracks_features in loader:
        with torch.no_grad():
            tracks_embeds = model(tracks_features)
            for track_id, track_embed in zip(tracks_ids, tracks_embeds):
                embeds[track_id] = track_embed.cpu().numpy()
    return embeds


def save_submission(submission, submission_path):
    with open(submission_path, 'w') as f:
        for query_trackid, result in submission.items():
            f.write("{}\t{}\n".format(query_trackid, " ".join(map(str, result))))


def get_ranked_list(embeds1, embeds2, embeds3, embeds4, embeds5, top_size, device):
    i_to_track_id = []

    embs1 = []
    for track_id, track_embed in embeds1.items():
        i_to_track_id.append(track_id)
        embs1.append(track_embed)

    embs2 = [embeds2[track_id] for track_id in i_to_track_id]
    embs3 = [embeds3[track_id] for track_id in i_to_track_id]
    embs4 = [embeds4[track_id] for track_id in i_to_track_id]
    embs5 = [embeds5[track_id] for track_id in i_to_track_id]

    embs1 = torch.tensor(np.array(embs1)).to(device)
    embs2 = torch.tensor(np.array(embs2)).to(device)
    embs3 = torch.tensor(np.array(embs3)).to(device)
    embs4 = torch.tensor(np.array(embs4)).to(device)
    embs5 = torch.tensor(np.array(embs5)).to(device)

    embs1 = F.normalize(embs1, dim=-1)
    embs2 = F.normalize(embs2, dim=-1)
    embs3 = F.normalize(embs3, dim=-1)
    embs4 = F.normalize(embs4, dim=-1)
    embs5 = F.normalize(embs5, dim=-1)

    embs = torch.cat((embs1, 0.8 * embs2, 0.55 * embs3, 0.49 * embs4, 0.41 * embs5), dim=-1)
    embs = F.normalize(embs, dim=-1)

    ranked_list = dict()

    for i, emb in enumerate(embs):
        scores = torch.matmul(embs, emb)
        assert scores.dim() == 1
        scores[i] = -1e10
        scores = torch.argsort(scores, descending=True)[:top_size].cpu().numpy()
        ranked_list[i_to_track_id[i]] = [i_to_track_id[j] for j in scores]

    return ranked_list


def main():
    parser = ArgumentParser(description='Simple naive baseline')
    parser.add_argument('--base-dir', dest='base_dir', action='store', required=True)
    args = parser.parse_args()

    # Seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    TESTSET_DIRNAME = 'test_features'
    TESTSET_META_FILENAME = 'test_meta.tsv'
    SUBMISSION_FILENAME = 'submission.txt'
    CHECKPOINT_FILENAME1 = 'best1.pt'
    CHECKPOINT_FILENAME2 = 'best2.pt'
    CHECKPOINT_FILENAME3 = 'best3.pt'
    CHECKPOINT_FILENAME4 = 'best4.pt'
    CHECKPOINT_FILENAME5 = 'best5.pt'
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cuda'

    BATCH_SIZE = 512

    TESTSET_PATH = os.path.join(args.base_dir, TESTSET_DIRNAME)
    TESTSET_META_PATH = os.path.join(args.base_dir, TESTSET_META_FILENAME)
    SUBMISSION_PATH = os.path.join(args.base_dir, SUBMISSION_FILENAME)
    CHECKPOINT_PATH1 = os.path.join(args.base_dir, CHECKPOINT_FILENAME1)
    CHECKPOINT_PATH2 = os.path.join(args.base_dir, CHECKPOINT_FILENAME2)
    CHECKPOINT_PATH3 = os.path.join(args.base_dir, CHECKPOINT_FILENAME3)
    CHECKPOINT_PATH4 = os.path.join(args.base_dir, CHECKPOINT_FILENAME4)
    CHECKPOINT_PATH5 = os.path.join(args.base_dir, CHECKPOINT_FILENAME5)

    test_meta_info = pd.read_csv(TESTSET_META_PATH, sep='\t')

    print("Loaded data")
    print("Test set size: {}".format(len(test_meta_info)))
    print()

    print("Submission")
    test_loader = TestLoader(FeaturesLoader(TESTSET_PATH, test_meta_info, device), batch_size=BATCH_SIZE)

    model1 = BasicNet1(1024).to(device)
    model1.load_state_dict(torch.load(CHECKPOINT_PATH1))

    model2 = BasicNet2(512).to(device)
    model2.load_state_dict(torch.load(CHECKPOINT_PATH2))

    model3 = BasicNet3(512).to(device)
    model3.load_state_dict(torch.load(CHECKPOINT_PATH3))

    model4 = BasicNet4(512).to(device)
    model4.load_state_dict(torch.load(CHECKPOINT_PATH4))

    model5 = BasicNet5(512).to(device)
    model5.load_state_dict(torch.load(CHECKPOINT_PATH5))

    embeds1 = inference(model1, test_loader)
    embeds2 = inference(model2, test_loader)
    embeds3 = inference(model3, test_loader)
    embeds4 = inference(model4, test_loader)
    embeds5 = inference(model5, test_loader)

    submission = get_ranked_list(embeds1, embeds2, embeds3, embeds4, embeds5, 100, device)
    save_submission(submission, SUBMISSION_PATH)


if __name__ == '__main__':
    main()
