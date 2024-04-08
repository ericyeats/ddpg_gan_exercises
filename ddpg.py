import gym
import numpy as np
import keras as K
import tensorflow as tf
from collections import deque
from argparse import ArgumentParser

MIN_ACT, MAX_ACT = -2, 2
STATE_SIZE = 3
ACT_SIZE = 1

mse_loss = K.losses.MeanSquaredError()

# the actor receives a state and outputs an action
def get_A_net_arch(h_dim: int) -> K.Sequential:
    return K.Sequential([
        K.layers.Dense(h_dim, activation='tanh'),
        K.layers.Dense(h_dim, activation='tanh'),
        K.layers.Dense(ACT_SIZE, activation='tanh')
    ])

# the  receives a tuple (state, action) and outputs an e
def get_Q_net_arch(h_dim: int) -> K.Sequential:
    return K.Sequential([
        K.layers.Dense(h_dim, activation='tanh'),
        K.layers.Dense(h_dim, activation='tanh'),
        K.layers.Dense(1, activation=None)
    ])

def A_net_out_act(out: tf.Tensor) -> tf.Tensor:
    assert tf.math.reduce_min(out) >= -1. and tf.math.reduce_max(out) <= 1. # check bounds
    return 2. * out

def Q_net_out_act(out: tf.Tensor):
    return out # no activation
    # assert tf.math.reduce_min(out) >= -1. and tf.math.reduce_max(out) <= 1. # check bounds
    # return (out + 1.) * (-16.2736044 / 2.)


def update_target_network(online_model: K.Model, target_model: K.Model, rho: float) -> None:
    scaled_target_weights = [tw*rho for tw in target_model.get_weights()]
    scaled_online_weights = [(1. - rho) * ow for ow in online_model.get_weights()]
    updated_weights = [tw + ow for tw, ow in zip(scaled_target_weights, scaled_online_weights)]
    target_model.set_weights(updated_weights)

def mean_squared_bellman_error(sarsd: tf.Tensor, online_Q_net: K.Model, target_A_net: K.Model, target_Q_net: K.Model, discount: float = 1.):
    # compute the bellman value using target Q, A nets
    sa = tf.slice(sarsd, begin=[0, 0], size=[-1, STATE_SIZE+ACT_SIZE])
    r = tf.slice(sarsd, begin=[0, STATE_SIZE+ACT_SIZE], size=[-1, 1])
    ns = tf.slice(sarsd, begin=[0, STATE_SIZE+ACT_SIZE+1], size=[-1, STATE_SIZE])
    d = tf.slice(sarsd, begin=[0, STATE_SIZE+ACT_SIZE+1+STATE_SIZE], size=[-1, 1])
    na = target_A_net(ns)
    nsa = tf.concat([ns, na], axis=1)
    y_bell = r + discount*(1. - d)*target_Q_net(nsa)
    pred_bell = online_Q_net(sa)
    msbe_loss = mse_loss(y_bell, pred_bell)
    return msbe_loss



class DeepDeterministicPolicyGradient:

    def __init__(self, 
            A_net: K.Model,
            Q_net: K.Model,
            target_A_net: K.Model,
            target_Q_net: K.Model,
            replay_buffer_size: int
            ):
        super().__init__()
        self.A_net = A_net
        self.Q_net = Q_net
        self.target_A_net = target_A_net
        self.target_Q_net = target_Q_net
        update_target_network(self.A_net, self.target_A_net, 0.) # copy the weights
        update_target_network(self.Q_net, self.target_Q_net, 0.) # copy the weights
        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer = deque(maxlen=replay_buffer_size)

    def update_target_networks(self, rho: float):
        update_target_network(self.A_net, self.target_A_net, rho)
        update_target_network(self.Q_net, self.target_Q_net, rho)

    def store_replay_buffer(self, state, action, reward, next_state, done):
        elem = np.concatenate([state, action, [reward,], next_state, [0. if done else 1.,]], axis=0)
        self.replay_buffer.append(elem[None, :])

    def sample_batch(self, batch_size: int) -> tf.Tensor:
        np_buffer = np.concatenate(self.replay_buffer, axis=0)
        inds = np.arange(np_buffer.shape[0], dtype=int)
        np.random.shuffle(inds)
        inds = inds[:batch_size]
        batch = tf.convert_to_tensor(np_buffer[inds], dtype=tf.float32)
        return batch

    def A_net_predict(self, s: tf.Tensor, noise_scale: float = 0., target=False) -> tf.Tensor:
        A_net = self.target_A_net if target else self.A_net
        pred = A_net_out_act(A_net(s))
        pred = tf.random.normal(pred.shape) * noise_scale + pred
        return tf.clip_by_value(pred, MIN_ACT, MAX_ACT) # could truncate gradient if used during backprop. careful
    
    def Q_net_predict(self, sa: tf.Tensor, target=False) -> tf.Tensor:
        Q_net = self.target_Q_net if target else self.Q_net
        pred = Q_net_out_act(Q_net(sa))
        return pred
    





if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('--h_dim', type=int, default=64, help="hidden layer size")
    parser.add_argument('--gravity', type=float, default=10., help="acceleration of gravity for Pendulum-v1 environment")
    parser.add_argument('--gamma', type=float, default=0.2, help='temporal reward discounting hyperparameter. should be in [0, 1]')
    parser.add_argument('--replay_buffer_size', type=int, default=256, help='replay buffer size to approx iid')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--target_rho', type=float, default=0.95, help='target network update momentum. tn = tn*rho + on(1-rho)')
    parser.add_argument('--exploration_scale', type=float, default=1., help='scale of Gaussian noise to add to actions during learning')
    parser.add_argument('--n_iters', type=int, default=2000, help='number of training iterations')
    parser.add_argument('--n_eval_iters', type=int, default=100, help='number of iters for evaluation')
    parser.add_argument('--n_updates_per_iter', type=int, default=1, help='A, Q updates per training iteration')


    args = parser.parse_args()
    
    env = gym.make('Pendulum-v1', g=args.gravity)
    print("Observation Space: ", env.observation_space)
    print("Action Space: ", env.action_space)

    # initalize online and target networks
    A_net = get_A_net_arch(args.h_dim)
    Q_net = get_Q_net_arch(args.h_dim)
    target_A_net = get_A_net_arch(args.h_dim)
    target_Q_net = get_Q_net_arch(args.h_dim)
    
    # create the DDPG
    ddpg = DeepDeterministicPolicyGradient(
        A_net=A_net, 
        Q_net=Q_net, 
        target_A_net=target_A_net, 
        target_Q_net=target_Q_net,
        replay_buffer_size=args.replay_buffer_size
        )
    
    A_opt = K.optimizers.Adam(1e-3)
    Q_opt = K.optimizers.Adam(1e-4)

    msbe_loss_tracker = []
    policy_loss_tracker = []

    # training loop
    obs = env.reset()[0]
    act = env.action_space.sample()
    print("OBS: ", obs)
    print("ACT: ", act)
    for iteration in range(args.n_iters):

        # observe the state
        # select an action (with noise). clip
        noisy_act = ddpg.A_net_predict(obs[None, :], noise_scale=args.exploration_scale)[0]

        # observe next state, reward, and done signal
        tup = env.step(noisy_act)
        next_obs, reward, done, truncated, info = tup
        # print(next_obs, reward, done, truncated, info)
        # store in the replay buffer
        ddpg.store_replay_buffer(obs, noisy_act, reward, next_obs, done or truncated)

        print("Iter: {}\tState: {}\tAct: {}\tNextState: {}".format(iteration, obs, noisy_act, next_obs))

        # if terminal state reached, reset the environment
        if done:
            obs = env.reset()
        else:
            obs = next_obs

        if len(ddpg.replay_buffer) >= args.replay_buffer_size - 1: # begin learning

            for update_iter in range(args.n_updates_per_iter):

                # sample a batch of transitions
                batch = ddpg.sample_batch(args.batch_size)

                # update the Q function using MBSE. store value of MBSE
                with tf.GradientTape() as tape:
                    msbe_loss = mean_squared_bellman_error(
                        batch,
                        lambda x: ddpg.Q_net_predict(x, target=False),
                        lambda x: ddpg.A_net_predict(x, target=True),
                        lambda x: ddpg.Q_net_predict(x, target=True),
                        args.gamma
                    )
                grads = tape.gradient(msbe_loss, ddpg.Q_net.trainable_weights)
                Q_opt.apply(grads, ddpg.Q_net.trainable_weights)
                msbe_loss_tracker.append(msbe_loss)

                # update the policy by ascending Q. store value of Q
                with tf.GradientTape() as tape:
                    s = tf.slice(batch, begin=[0,0], size=[-1, STATE_SIZE])
                    a = ddpg.A_net_predict(s)
                    sa = tf.concat([s, a], axis=1)
                    policy_loss = -tf.reduce_mean(ddpg.Q_net_predict(sa)) # ascend the Q function
                grads = tape.gradient(policy_loss, ddpg.A_net.trainable_weights)
                A_opt.apply(grads, ddpg.A_net.trainable_weights)
                policy_loss_tracker.append(policy_loss)

                # update target networks
                ddpg.update_target_networks(args.target_rho)

    import matplotlib.pyplot as plt
    # visualize losses
    plt.figure()

    plt.plot(msbe_loss_tracker, linewidth=3, label="MSBE Loss")
    plt.plot(policy_loss_tracker, linewidth=3, label="Policy Loss")
    plt.legend()
    plt.xlabel("Training Iteration")
    plt.ylabel("Loss Value")
    plt.grid()
    plt.savefig("./RL_losses.png")


    control_tracker = []
    reward_tracker = []
    angle_tracker = []
    vel_tracker = []
    # visualize the state, control signals, and reward with time for one trial
    obs = env.reset()[0]
    for iteration in range(args.n_eval_iters):
        x, y, vel = obs
        angle = np.arctan2(y, x)
        angle_tracker.append(angle)
        vel_tracker.append(vel)
        # observe the state
        # select an action (without noise). clip
        act = ddpg.A_net_predict(obs[None, :])[0]
        # observe next state, reward, and done signal
        tup = env.step(act)
        next_obs, reward, done, truncated, info = tup
        # if terminal state reached, reset the environment
        if done:
            obs = env.reset()
        else:
            obs = next_obs

        control_tracker.append(act)
        reward_tracker.append(reward)

    plt.figure()

    plt.plot(control_tracker, linewidth=3, label="Control Signal")
    plt.plot(reward_tracker, linewidth=3, label="Reward")
    plt.plot(angle_tracker, linewidth=3, label='Angle (Radians)')
    plt.plot(vel_tracker, linewidth=3, label='Angular Velocity (Rad/s)')
    plt.legend()
    plt.xlabel("Eval Iteration")
    plt.ylabel("Value")
    plt.grid()
    plt.savefig("./RL_eval.png")
    
