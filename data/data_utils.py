import numpy as np
from argparse import ArgumentParser

from typing import List, Tuple

def density_plots(arrs: List[np.ndarray], sig: float = 0.1, n_t: int = 1000) -> Tuple[np.ndarray, List[np.ndarray]]:
    # create bounds for the plot
    t_min = min([np.min(a) for a in arrs])
    t_max = max([np.max(a) for a in arrs])

    t_min -= sig*5
    t_max += sig*5

    gauss_density = lambda x, mu, sig: ((sig**2 * 2. * np.pi)**-0.5)*np.exp(-0.5 * ((x - mu)/sig)**2)

    t = np.linspace(t_min, t_max, n_t)

    out = []
    for a in arrs:
        a_dense = np.zeros_like(t)
        for i, t_val in enumerate(t):
            a_dense[i] = np.mean(gauss_density(t_val, a, sig)) # uniform weighting
        out.append(a_dense)
    return t, out

def kde_jsd(dist_a: np.ndarray, dist_b: np.ndarray, sig: float, n_t: int, eps=1e-5) -> float:
    t, (kde_a, kde_b) = density_plots([dist_a, dist_b], sig, n_t)

    # use the KDEs to estimate the KL divergences between the distributions
    kl_ab = np.sum(kde_a*(np.log(kde_a+eps) - np.log(kde_b+eps))) * (t[1] - t[0]) # dt
    kl_ba = np.sum(kde_b*(np.log(kde_b+eps) - np.log(kde_a+eps))) * (t[1] - t[0])

    jsd = (kl_ab + kl_ba) / 2.
    return jsd


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--source_path", type=str, default='eICU_age.npy')
    parser.add_argument("--target_name", type=str, default='eICU_age_proc')

    parser.add_argument("--val_split", type=float, default=0.2, help="fraction of data reserved for validation")
    parser.add_argument("--seed", type=int, default=None, help="seed for shuffling the data before splitting")

    parser.add_argument("--plot", action="store_true", help="create and save KDE plots of the train/val data")

    args = parser.parse_args()


    # set the seed
    np.random.seed(args.seed)

    # load the data
    data = np.load(args.source_path)

    data = data.astype(np.float)

    np.random.shuffle(data)

    # create the splits
    train_len = int(len(data)*(1.-args.val_split))
    data_train, data_val = data[:train_len], data[train_len:]

    # save the splits
    np.save(args.target_name + "_train.npy", data_train)
    np.save(args.target_name + "_val.npy", data_val)

    if args.plot:
        import matplotlib.pyplot as plt

        alpha = 0.5
        sig = 2.
        
        # create some KDEs of the data
        t, (train_kde, val_kde) = density_plots([data_train, data_val], sig=sig, n_t=1000)
        jsd = kde_jsd(data_train, data_val, sig=sig, n_t=1000)
        print("Train/Val Jensen-Shannon Divergence Estimate: {:1.5f}".format(jsd))

        plt.figure()
        plt.plot(t, train_kde, linewidth=3, color='tab:blue', label="Train Data N={}".format(data_train.shape[0]))
        plt.fill_between(t, np.zeros_like(train_kde), train_kde, color='tab:blue', alpha=alpha)
        plt.plot(t, val_kde, linewidth=3, color='tab:orange', label="Val Data N={}".format(data_val.shape[0]))
        plt.fill_between(t, np.zeros_like(val_kde), val_kde, color='tab:orange', alpha=alpha)

        plt.grid()
        plt.legend()

        plt.xlabel("Age (Years)")
        plt.ylabel(r"Kernel Density Estimate ($\sigma$={:1.2f})".format(sig))
        plt.title("KDE of {}".format(args.source_path))

        plt.savefig("./kde.png")

