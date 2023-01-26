import numpy as np
import pandas as pd
import os
from argparse import ArgumentParser
import torch
import random
import torch.nn.functional as F

from train1 import FeaturesLoader as FeaturesLoader1
from train2 import FeaturesLoader as FeaturesLoader2
from train3 import FeaturesLoader as FeaturesLoader3
from train4 import FeaturesLoader as FeaturesLoader4
from train5 import FeaturesLoader as FeaturesLoader5
from train6 import FeaturesLoader as FeaturesLoader6

from train1 import TestLoader as TestLoader1
from train2 import TestLoader as TestLoader2
from train3 import TestLoader as TestLoader3
from train4 import TestLoader as TestLoader4
from train5 import TestLoader as TestLoader5
from train6 import TestLoader as TestLoader6

from train1 import BasicNet as BasicNet1
from train2 import BasicNet as BasicNet2
from train3 import BasicNet as BasicNet3
from train4 import BasicNet as BasicNet4
from train5 import BasicNet as BasicNet5
from train6 import BasicNet as BasicNet6

from train1 import inference as inference1
from train2 import inference as inference2
from train3 import inference as inference3
from train4 import inference as inference4
from train5 import inference as inference5
from train6 import inference as inference6


def save_submission(submission, submission_path):
    with open(submission_path, 'w') as f:
        for query_trackid, result in submission.items():
            f.write("{}\t{}\n".format(query_trackid, " ".join(map(str, result))))


def get_ranked_list(embeds1, embeds2, embeds3, embeds4, embeds5, embeds6, top_size, device):
    i_to_track_id = []

    embs1 = []
    for track_id, track_embed in embeds1.items():
        i_to_track_id.append(track_id)
        embs1.append(track_embed)

    embs2 = [embeds2[track_id] for track_id in i_to_track_id]
    embs3 = [embeds3[track_id] for track_id in i_to_track_id]
    embs4 = [embeds4[track_id] for track_id in i_to_track_id]
    embs5 = [embeds5[track_id] for track_id in i_to_track_id]
    embs6 = [embeds6[track_id] for track_id in i_to_track_id]

    embs1 = torch.tensor(np.array(embs1)).to(device)
    embs2 = torch.tensor(np.array(embs2)).to(device)
    embs3 = torch.tensor(np.array(embs3)).to(device)
    embs4 = torch.tensor(np.array(embs4)).to(device)
    embs5 = torch.tensor(np.array(embs5)).to(device)
    embs6 = torch.tensor(np.array(embs6)).to(device)

    embs1 = F.normalize(embs1, dim=-1)
    embs2 = F.normalize(embs2, dim=-1)
    embs3 = F.normalize(embs3, dim=-1)
    embs4 = F.normalize(embs4, dim=-1)
    embs5 = F.normalize(embs5, dim=-1)
    embs6 = F.normalize(embs6, dim=-1)

    embs = torch.cat((embs1, 1.19 * embs2, 0.81 * embs3, 0.8 * embs4, 0.71 * embs5, 0.53 * embs6), dim=-1)
    embs = F.normalize(embs, dim=-1)

    pp_weights = [25.0, 8.0, 5.0, 3.0, 2.0, 1.0]
    pp_weights_sum = sum(pp_weights)
    pp_weights = [x / pp_weights_sum for x in pp_weights]

    new_embs = torch.zeros_like(embs)

    for i, emb in enumerate(embs):
        scores = torch.matmul(embs, emb)
        scores[i] = -1e10
        scores = torch.argsort(scores, descending=True)[:len(pp_weights) - 1].cpu().numpy()
        new_emb = pp_weights[0] * emb
        for j in range(1, len(pp_weights)):
            new_emb += pp_weights[j] * embs[scores[j - 1]]
        new_embs[i] = new_emb

    embs = new_embs
    embs = F.normalize(embs, dim=-1)

    ranked_list = dict()

    for i, emb in enumerate(embs):
        scores = torch.matmul(embs, emb)
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
    CHECKPOINT_FILENAME6 = 'best6.pt'
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
    CHECKPOINT_PATH6 = os.path.join(args.base_dir, CHECKPOINT_FILENAME6)

    test_meta_info = pd.read_csv(TESTSET_META_PATH, sep='\t')

    print("Loaded data")
    print("Test set size: {}".format(len(test_meta_info)))
    print()

    print("Submission")
    test_loader1 = TestLoader1(FeaturesLoader1(TESTSET_PATH, test_meta_info, device), batch_size=BATCH_SIZE)
    test_loader2 = TestLoader2(FeaturesLoader2(TESTSET_PATH, test_meta_info, device), batch_size=BATCH_SIZE)
    test_loader3 = TestLoader3(FeaturesLoader3(TESTSET_PATH, test_meta_info, device), batch_size=BATCH_SIZE)
    test_loader4 = TestLoader4(FeaturesLoader4(TESTSET_PATH, test_meta_info, device), batch_size=BATCH_SIZE)
    test_loader5 = TestLoader5(FeaturesLoader5(TESTSET_PATH, test_meta_info, device), batch_size=BATCH_SIZE)
    test_loader6 = TestLoader6(FeaturesLoader6(TESTSET_PATH, test_meta_info, device), batch_size=BATCH_SIZE)

    model1 = BasicNet1().to(device)
    model1.load_state_dict(torch.load(CHECKPOINT_PATH1))

    model2 = BasicNet2().to(device)
    model2.load_state_dict(torch.load(CHECKPOINT_PATH2))

    model3 = BasicNet3().to(device)
    model3.load_state_dict(torch.load(CHECKPOINT_PATH3))

    model4 = BasicNet4().to(device)
    model4.load_state_dict(torch.load(CHECKPOINT_PATH4))

    model5 = BasicNet5().to(device)
    model5.load_state_dict(torch.load(CHECKPOINT_PATH5))

    model6 = BasicNet6().to(device)
    model6.load_state_dict(torch.load(CHECKPOINT_PATH6))

    embeds1 = inference1(model1, test_loader1)
    embeds2 = inference2(model2, test_loader2)
    embeds3 = inference3(model3, test_loader3)
    embeds4 = inference4(model4, test_loader4)
    embeds5 = inference5(model5, test_loader5)
    embeds6 = inference6(model6, test_loader6)

    # import pickle
    # with open('embeds.pickle', 'wb') as f:
    #     pickle.dump((embeds1, embeds2, embeds3, embeds4, embeds5, embeds6), f)
    # with open('embeds.pickle', 'rb') as f:
    #     embeds1, embeds2, embeds3, embeds4, embeds5, embeds6 = pickle.load(f)

    submission = get_ranked_list(embeds1, embeds2, embeds3, embeds4, embeds5, embeds6, 100, device)
    save_submission(submission, SUBMISSION_PATH)


if __name__ == '__main__':
    main()
