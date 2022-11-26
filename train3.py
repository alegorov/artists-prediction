import numpy as np
import pandas as pd
import os
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from tqdm import tqdm
import random
import torch.nn.functional as F


# Data Loader

def train_val_split(dataset, val_size=0.2):  # Сплит по artistid
    artist_ids = dataset['artistid'].unique()
    train_artist_ids, val_artist_ids = train_test_split(artist_ids, test_size=val_size)
    trainset = dataset[dataset['artistid'].isin(train_artist_ids)].copy()
    valset = dataset[dataset['artistid'].isin(val_artist_ids)].copy()
    return trainset, valset


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


class TrainLoader:
    def __init__(self, features_loader, batch_size=256):
        self.features_loader = features_loader
        self.batch_size = batch_size

        data = {}

        for row in self.features_loader.meta_info.to_dict('records'):
            item = data.get(row['artistid'], [])
            item.append(row['trackid'])
            data[row['artistid']] = item

        self.data = [item[1] for item in data.items()]

    def __iter__(self):
        random.shuffle(self.data)

        track_ids = []
        ids = []

        for id, row in enumerate(self.data):
            for track_id in row:
                track_ids.append(track_id)
                ids.append(id)
            if len(track_ids) >= self.batch_size:
                x = self.features_loader.load_batch(track_ids)
                yield x, ids
                track_ids = []
                ids = []


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

        # Loss & Metrics


class NT_Xent(nn.Module):
    def __init__(self, temperature, negative_weight):
        super(NT_Xent, self).__init__()
        self.temperature = temperature
        self.negative_weight = negative_weight
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, z, ids):
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        N = len(ids)

        positive_mask = torch.zeros((N, N), dtype=bool)
        negative_mask = torch.zeros((N, N), dtype=bool)

        for i in range(N):
            for j in range(i + 1, N):
                if ids[i] == ids[j]:
                    positive_mask[i, j] = 1
                else:
                    negative_mask[i, j] = 1

        clip = 70.

        positive_sim = clip * torch.tanh((1. / clip) * sim[positive_mask])
        negative_sim = clip * torch.tanh((1. / clip) * sim[negative_mask])

        positive_norm = torch.exp(positive_sim).mean()
        negative_norm = torch.exp(negative_sim).mean()

        loss = torch.log(positive_norm + self.negative_weight * negative_norm) - positive_sim.mean()
        return loss


def get_ranked_list(embeds, top_size, device):
    i2track_id = []
    all_embs = []
    for track_id, track_embed in embeds.items():
        i2track_id.append(track_id)
        all_embs.append(track_embed)

    all_embs = torch.tensor(np.array(all_embs)).to(device)

    all_embs = F.normalize(all_embs, dim=-1)

    ranked_list = dict()

    for i, emb in enumerate(all_embs):
        scores = torch.matmul(all_embs, emb)
        assert scores.dim() == 1
        scores[i] = -10
        scores = torch.argsort(scores, descending=True)[:top_size].cpu().numpy()
        ranked_list[i2track_id[i]] = [i2track_id[j] for j in scores]

    return ranked_list


def position_discounter(position):
    return 1.0 / np.log2(position + 1)


def get_ideal_dcg(relevant_items_count, top_size):
    dcg = 0.0
    for result_indx in range(min(top_size, relevant_items_count)):
        position = result_indx + 1
        dcg += position_discounter(position)
    return dcg


def compute_dcg(query_trackid, ranked_list, track2artist_map, top_size):
    query_artistid = track2artist_map[query_trackid]
    dcg = 0.0
    for result_indx, result_trackid in enumerate(ranked_list[:top_size]):
        assert result_trackid != query_trackid
        position = result_indx + 1
        discounted_position = position_discounter(position)
        result_artistid = track2artist_map[result_trackid]
        if result_artistid == query_artistid:
            dcg += discounted_position
    return dcg


def eval_submission(submission, gt_meta_info, top_size=100):
    track2artist_map = gt_meta_info.set_index('trackid')['artistid'].to_dict()
    artist2tracks_map = gt_meta_info.groupby('artistid').agg(list)['trackid'].to_dict()
    ndcg_list = []
    for query_trackid in tqdm(submission.keys()):
        ranked_list = submission[query_trackid]
        query_artistid = track2artist_map[query_trackid]
        query_artist_tracks_count = len(artist2tracks_map[query_artistid])
        ideal_dcg = get_ideal_dcg(query_artist_tracks_count - 1, top_size=top_size)
        dcg = compute_dcg(query_trackid, ranked_list, track2artist_map, top_size=top_size)
        try:
            ndcg_list.append(dcg / ideal_dcg)
        except ZeroDivisionError:
            continue
    return np.mean(ndcg_list)


class BasicNet(nn.Module):
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


class SimCLR(nn.Module):
    def __init__(self, encoder, projection_dim):
        super().__init__()
        self.encoder = encoder
        self.n_features = encoder.output_features_size
        self.projection_dim = projection_dim
        if projection_dim:
            relu = nn.ReLU()
            self.projector = nn.Sequential(
                # relu,
                # nn.Linear(self.n_features, self.n_features, bias=False),
                relu,
                nn.Linear(self.n_features, self.projection_dim, bias=False),
            )

    def forward(self, x):
        h = self.encoder(x)

        if self.projection_dim:
            z = self.projector(h)
        else:
            z = h

        return h, z


def inference(model, loader):
    embeds = dict()
    model.eval()
    for tracks_ids, tracks_features in loader:
        with torch.no_grad():
            tracks_embeds = model(tracks_features)
            for track_id, track_embed in zip(tracks_ids, tracks_embeds):
                embeds[track_id] = track_embed.cpu().numpy()
    return embeds


def train(model, train_loader, val_loader, valset_meta, optimizer, criterion, num_epochs, checkpoint_path, device,
          top_size=100):
    max_ndcg = None
    for epoch in range(num_epochs):
        model.train()
        for x, ids in tqdm(train_loader):
            optimizer.zero_grad()
            h, z = model(x)
            loss = criterion(z, ids)
            loss.backward()
            optimizer.step()
            print("Epoch {}/{}".format(epoch + 1, num_epochs))
            print("loss: {}".format(loss))
            print()

        model.eval()
        with torch.no_grad():
            model_encoder = model.encoder
            embeds_encoder = inference(model_encoder, val_loader)
            ranked_list_encoder = get_ranked_list(embeds_encoder, top_size, device)
            val_ndcg_encoder = eval_submission(ranked_list_encoder, valset_meta)

            print("Validation nDCG on epoch {}".format(epoch + 1))
            print("Encoder - {}".format(val_ndcg_encoder))

            if model.projection_dim:
                model_projector = nn.Sequential(model.encoder, model.projector)
                embeds_projector = inference(model_projector, val_loader)
                ranked_list_projector = get_ranked_list(embeds_projector, top_size, device)
                val_ndcg_projector = eval_submission(ranked_list_projector, valset_meta)
                print("Projector - {}".format(val_ndcg_projector))

            if (max_ndcg is None) or (val_ndcg_encoder > max_ndcg):
                max_ndcg = val_ndcg_encoder
                torch.save(model_encoder.state_dict(), checkpoint_path)
                with open(checkpoint_path + '.txt', 'w', encoding='utf-8') as text_file:
                    print("Validation nDCG on epoch {}".format(epoch + 1), file=text_file)
                    print("Encoder - {}".format(val_ndcg_encoder), file=text_file)
                    if model.projection_dim:
                        print("Projector - {}".format(val_ndcg_projector), file=text_file)


def save_submission(submission, submission_path):
    with open(submission_path, 'w') as f:
        for query_trackid, result in submission.items():
            f.write("{}\t{}\n".format(query_trackid, " ".join(map(str, result))))


def main():
    parser = ArgumentParser(description='Simple naive baseline')
    parser.add_argument('--base-dir', dest='base_dir', action='store', required=True)
    parser.add_argument('--is-train', dest='is_train', action='store', required=True)
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

    TRAINSET_DIRNAME = 'train_features'
    TESTSET_DIRNAME = 'test_features'
    TRAINSET_META_FILENAME = 'train_meta.tsv'
    TESTSET_META_FILENAME = 'test_meta.tsv'
    SUBMISSION_FILENAME = 'submission.txt'
    MODEL_FILENAME = 'model.pt'
    CHECKPOINT_FILENAME = 'best3.pt'
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cuda'

    BATCH_SIZE = 460
    N_CHANNELS = 512
    PROJECTION_DIM = 0
    NUM_EPOCHS = 48
    LR = 1e-4
    TEMPERATURE = 0.011
    NEGATIVE_WEIGHT = 930.

    TRAINSET_PATH = os.path.join(args.base_dir, TRAINSET_DIRNAME)
    TESTSET_PATH = os.path.join(args.base_dir, TESTSET_DIRNAME)
    TRAINSET_META_PATH = os.path.join(args.base_dir, TRAINSET_META_FILENAME)
    TESTSET_META_PATH = os.path.join(args.base_dir, TESTSET_META_FILENAME)
    SUBMISSION_PATH = os.path.join(args.base_dir, SUBMISSION_FILENAME)
    MODEL_PATH = os.path.join(args.base_dir, MODEL_FILENAME)
    CHECKPOINT_PATH = os.path.join(args.base_dir, CHECKPOINT_FILENAME)

    sim_clr = SimCLR(
        encoder=BasicNet(N_CHANNELS),
        projection_dim=PROJECTION_DIM
    ).to(device)

    train_meta_info = pd.read_csv(TRAINSET_META_PATH, sep='\t')
    test_meta_info = pd.read_csv(TESTSET_META_PATH, sep='\t')
    train_meta_info, validation_meta_info = train_val_split(train_meta_info, val_size=0.1)

    print("Loaded data")
    print("Train set size: {}".format(len(train_meta_info)))
    print("Validation set size: {}".format(len(validation_meta_info)))
    print("Test set size: {}".format(len(test_meta_info)))
    print()

    if int(args.is_train):
        print("Train")
        train(
            model=sim_clr,
            train_loader=TrainLoader(FeaturesLoader(TRAINSET_PATH, train_meta_info, device), batch_size=BATCH_SIZE),
            val_loader=TestLoader(FeaturesLoader(TRAINSET_PATH, validation_meta_info, device), batch_size=BATCH_SIZE),
            valset_meta=validation_meta_info,
            optimizer=torch.optim.Adam(sim_clr.parameters(), lr=LR),
            criterion=NT_Xent(temperature=TEMPERATURE, negative_weight=NEGATIVE_WEIGHT),
            num_epochs=NUM_EPOCHS,
            checkpoint_path=CHECKPOINT_PATH,
            device=device
        )
        torch.save(sim_clr.state_dict(), MODEL_PATH)

    print("Submission")
    test_loader = TestLoader(FeaturesLoader(TESTSET_PATH, test_meta_info, device), batch_size=BATCH_SIZE)
    model = BasicNet(N_CHANNELS).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    embeds = inference(model, test_loader)
    submission = get_ranked_list(embeds, 100, device)
    save_submission(submission, SUBMISSION_PATH)


if __name__ == '__main__':
    main()
