import time
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras as K
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from data.data_utils import density_plots, kde_jsd

def get_discriminator(h_dim: int = 64):
    return K.Sequential([
        K.layers.Dense(h_dim, activation='elu'),
        K.layers.Dense(h_dim, activation='elu'),
        K.layers.Dense(1, activation=None)
    ])

def get_generator(h_dim: int = 64):
    return K.Sequential([
        K.layers.Dense(h_dim, activation='elu'),
        K.layers.Dense(h_dim, activation='elu'),
        K.layers.Dense(1, activation=None)
    ])


if __name__ == "__main__":

    parser = ArgumentParser()

    # general args
    parser.add_argument("--train_data_path", type=str, default='./data/eICU_age_proc_train.npy', help="path to train dataset")
    parser.add_argument("--val_data_path", type=str, default='./data/eICU_age_proc_val.npy', help="path to val dataset")
    parser.add_argument("--h_dim", type=int, default=64, help="dimension of the hidden layers")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--n_iters", type=int, default=10000, help="number of training iterations")
    parser.add_argument("--kde_sigma", type=float, default=2., help="sigma parameter for KDE")

    args = parser.parse_args()

    # get the train/val data
    train_data = np.load(args.train_data_path)
    val_data = np.load(args.val_data_path)

    train_mean = np.mean(train_data)
    train_std = np.std(train_data)

    # instantiate the models
    disc = get_discriminator(h_dim=args.h_dim*2)
    gen = get_generator(h_dim=args.h_dim)

    disc_opt = K.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.99, weight_decay=1e-4)
    gen_opt = K.optimizers.Adam(learning_rate=2e-5, beta_1=0.9, beta_2=0.99, weight_decay=1e-4)

    loss_fn = K.losses.BinaryCrossentropy(from_logits=True)

    data_tform = lambda arr: (arr - train_mean) / train_std
    data_inv_tform = lambda arr: arr * train_std + train_mean

    @tf.function
    def train_step():
        # extract a batch of data
        inds = np.arange(train_data.shape[0], dtype=int)
        np.random.shuffle(inds)
        inds = inds[:args.batch_size]
        data_batch = np.reshape(train_data[inds], (-1, 1)) # (batch_size, n_feat)
        data_batch = data_tform(data_batch)
        data_batch = tf.convert_to_tensor(data_batch, dtype=tf.float32)

        # create some generated data
        noise_batch = tf.random.normal(shape=(args.batch_size, args.h_dim)) # (batch_size, args.h_dim)
        gen_out = gen(noise_batch)

        combined_batch = tf.concat([data_batch, gen_out], axis=0)
        combined_targets = tf.concat([tf.zeros((args.batch_size, 1)), tf.ones((args.batch_size, 1))], axis=0)
        combined_targets += 0.05 * tf.random.uniform(combined_targets.shape) # recommended trick

        # train the discriminator
        with tf.GradientTape() as tape:
            disc_out = disc(combined_batch)
            disc_loss = loss_fn(combined_targets, disc_out)

        grads = tape.gradient(disc_loss, disc.trainable_weights)
        disc_opt.apply(grads, disc.trainable_weights) # discriminator training step


        noise_batch = tf.random.normal(shape=(args.batch_size, args.h_dim))

        with tf.GradientTape() as tape:
            gen_out = gen(noise_batch)
            disc_out = disc(gen_out)
            target = tf.zeros((args.batch_size, 1)) # want to fool discriminator. this trick provides stronger learning signal
            gen_loss = loss_fn(target, disc_out)

        grads = tape.gradient(gen_loss, gen.trainable_weights)
        gen_opt.apply(grads, gen.trainable_weights) # generator training step

        return disc_loss, gen_loss


    disc_losses = []
    gen_losses = []
    val_losses = []

    for iteration in range(args.n_iters):
        disc_loss, gen_loss = train_step()
        disc_losses.append(disc_loss)
        gen_losses.append(gen_loss)

        if (iteration+1) % 100 == 0:
            noise_samples = tf.random.normal((val_data.shape[0], args.h_dim))
            gen_out = gen(noise_samples)
            gen_out = data_inv_tform(tf.squeeze(gen_out).numpy())
            val_loss = kde_jsd(gen_out, val_data, sig=args.kde_sigma, n_t=1000)
            val_losses.append(val_loss)
            print("Iter: {}\t Disc Loss: {:1.2f}\t Gen Loss: {:1.2f}\t Val JSD: {:1.3f}".format(iteration + 1, disc_loss, gen_loss, val_loss))

    
    # plot all the losses
    plt.figure()
    plt.plot(disc_losses, linewidth=3, label="Discriminator Loss")
    plt.plot(gen_losses, linewidth=3, label="Generator Loss")
    
    plt.grid()
    plt.xlabel("Training Iteration")
    plt.ylabel("Loss Value")
    plt.title("Training Losses")
    plt.legend()
    plt.savefig("./train_losses.png")

    plt.figure()
    plt.plot(val_losses, linewidth=3, label="Validation JSD (Generator)")
    plt.grid()
    plt.xlabel("Validation Iteration")
    plt.ylabel("Val Metric")
    plt.title("JSD of Generator Output with Validation Set")
    plt.savefig("./val_metric.png")

    # create a final KDE of the Generator output vs the val distribution
    alpha = 0.5
    noise_samples = tf.random.normal((val_data.shape[0], args.h_dim))
    gen_out = gen(noise_samples)
    gen_out = data_inv_tform(tf.squeeze(gen_out).numpy())
    t, (gen_kde, val_kde) = density_plots([gen_out, val_data], args.kde_sigma, 1000)
    jsd = kde_jsd(gen_out, val_data, sig=args.kde_sigma, n_t=1000)
    print("JSD: {:1.4f}".format(jsd))

    plt.figure()
    plt.plot(t, val_kde, linewidth=3, color='tab:orange', label="Val Data N={}".format(val_data.shape[0]))
    plt.fill_between(t, np.zeros_like(val_kde), val_kde, color='tab:orange', alpha=alpha)
    plt.plot(t, gen_kde, linewidth=3, color='tab:purple', label="Gen Data N={}".format(gen_out.shape[0]))
    plt.fill_between(t, np.zeros_like(gen_kde), gen_kde, color='tab:purple', alpha=alpha)

    plt.grid()
    plt.legend()

    plt.xlabel("Age (Years)")
    plt.ylabel(r"Kernel Density Estimate ($\sigma$={:1.2f})".format(args.kde_sigma))
    plt.title("KDE of Generator & Validation: JSD = {:1.4f}".format(jsd))
    plt.savefig("./gen_val_kde.png")


    


