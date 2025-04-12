import os
import numpy as np
import torch

# --- CONFIG ---
PARENT_DIR = '.'  # where your seed subfolders (named '0','1',... etc) live
CLASSIFIERS = ['knn', 'svm', 'decision_tree', 'random_forest']
ROOT_SELECT_RATE = 'select_rate.pt'  # the one saved at the script root

def find_seed_dirs(parent):
    return sorted(
        [d for d in os.listdir(parent)
         if os.path.isdir(os.path.join(parent, d)) and d.isdigit()],
        key=int
    )

def load_record(seed_dir):
    rec_path = os.path.join(seed_dir, 'record.npy')
    if not os.path.exists(rec_path):
        raise FileNotFoundError(f"Missing {rec_path}")
    return np.load(rec_path)

def aggregate(parent_dir):
    seed_dirs = find_seed_dirs(parent_dir)
    acc_sums = {clf: 0.0 for clf in CLASSIFIERS}
    ks = {clf: [] for clf in CLASSIFIERS}

    for sd in seed_dirs:
        rec = load_record(os.path.join(parent_dir, sd))
        for i, clf in enumerate(CLASSIFIERS):
            acc = rec[2*i]
            k   = int(rec[2*i + 1])
            acc_sums[clf] += acc
            ks[clf].append(k)

    n = len(seed_dirs)
    avg_accs = {clf: acc_sums[clf]/n for clf in CLASSIFIERS}
    return avg_accs, ks

def consensus_k(k_list):
    vals, counts = np.unique(k_list, return_counts=True)
    return int(vals[np.argmax(counts)])

def main():
    # 1) Aggregate
    avg_accs, ks = aggregate(PARENT_DIR)

    # 2) Best classifier
    best_clf = max(avg_accs, key=avg_accs.get)
    best_avg_acc = avg_accs[best_clf]

    # 3) Consensus k
    best_k = consensus_k(ks[best_clf])

    # 4) Load select_rate.pt and extract top-k
    if not os.path.exists(ROOT_SELECT_RATE):
        raise FileNotFoundError(f"Cannot find {ROOT_SELECT_RATE}")
    sr_tensor = torch.load(ROOT_SELECT_RATE, weights_only=True)
    sr = sr_tensor.detach().cpu().numpy()
    top_indices = np.argsort(-sr)[:best_k].tolist()

    # 5) Print
    print("=== Summary Across Seeds ===")
    print(f"Best Classifier    : {best_clf}")
    print(f"Average Accuracy   : {best_avg_acc:.4f}")
    print(f"Consensus k        : {best_k}")
    print(f"Top-{best_k} feature indices: {top_indices}")
def hi(_):
    pass
if __name__ == '__main__':
    main()
