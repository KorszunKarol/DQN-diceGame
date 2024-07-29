from __future__ import absolute_import, division, print_function

import base64
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay
import reverb
import tensorflow as tf
from tf_agents.specs import array_spec
from tf_agents.environments import utils
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.policies import policy_saver
from game import Game

from tf_agents.metrics import py_metric
from tf_agents.metrics import py_metrics
from tf_agents.metrics import tf_metrics
from tf_agents.metrics import tf_py_metric
from tf_agents.drivers import dynamic_episode_driver
import os

root_dir = os.path.join(os.getcwd(), "logs")
train_dir = os.path.join(root_dir, "train")
train_summary_writer = tf.summary.create_file_writer(train_dir, flush_millis=10000)
train_summary_writer.set_as_default()

os.environ["TF_USE_LEGACY_KERAS"] = "1"











def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode="fan_in", distribution="truncated_normal"
        ),
    )


def DQN(env):
    fc_layer_params = (200, 100)
    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03
        ),
        bias_initializer=tf.keras.initializers.Constant(-0.2),
    )
    q_net = sequential.Sequential(dense_layers + [q_values_layer])
    return q_net


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0

    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward

        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def create_replay_buffer(agent, replay_buffer_max_length):
    table_name = "uniform_table"
    replay_buffer_signature = tensor_spec.from_spec(agent.collect_data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)

    table = reverb.Table(
        table_name,
        max_size=replay_buffer_max_length,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature,
    )

    reverb_server = reverb.Server([table])

    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
        agent.collect_data_spec,
        table_name=table_name,
        sequence_length=2,
        local_server=reverb_server,
    )

    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
        replay_buffer.py_client, table_name, sequence_length=2
    )

    return replay_buffer, rb_observer


def create_agents(train_env, q_net, optimizer, train_step_counter):
    action_spec = array_spec.BoundedArraySpec(
        shape=(),
        dtype=np.int64,
        minimum=train_env.action_spec().minimum,
        maximum=train_env.action_spec().maximum,
        name=train_env.action_spec().name,
    )

    agent = dqn_agent.DqnAgent(
        time_step_spec=train_env.time_step_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter,
        action_spec=train_env.action_spec(),
    )
    eval_policy = agent.policy
    collect_policy = agent.collect_policy

    agent.initialize()
    return agent


def main():

    num_iterations = 5000
    initial_collect_steps = (
        100
    )
    collect_steps_per_iteration = 20
    replay_buffer_max_length = (
        1000000
    )
    batch_size = 64
    learning_rate = 1e-3
    log_interval = 100
    num_eval_episodes = 100
    eval_interval = 1000

    env = Game(target_score=50)
    print(utils.validate_py_environment(env, episodes=1))

    print(env.time_step_spec().observation)

    train_py_env = Game(target_score=50)
    eval_py_env = Game(target_score=50)
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    train_env.reset()
    eval_env.reset()

    q_net = DQN(train_env)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)

    time_step = eval_env.reset()
    random_policy = random_tf_policy.RandomTFPolicy(
        time_step_spec=eval_env.time_step_spec(), action_spec=eval_env.action_spec()
    )





    time_step = train_env.reset()
    random_policy.action(time_step)
    agent = create_agents(
        train_env=train_env,
        q_net=q_net,
        optimizer=optimizer,
        train_step_counter=train_step_counter,
    )
    print(agent.collect_data_spec._fields)

    replay_buffer, rb_observer = create_replay_buffer(agent, replay_buffer_max_length)

    py_driver.PyDriver(
        env,
        py_tf_eager_policy.PyTFEagerPolicy(random_policy, use_tf_function=True),
        [rb_observer],
        max_steps=initial_collect_steps,
    ).run(train_py_env.reset())


    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=64,
        num_steps=2,
    ).prefetch(3)

    iterator = iter(dataset)
    print(iterator)
    metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]







    agent.train = common.function(agent.train)


    agent.train_step_counter.assign(0)

    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]


    time_step = train_py_env.reset()


    collect_driver = py_driver.PyDriver(
        env,
        py_tf_eager_policy.PyTFEagerPolicy(agent.collect_policy, use_tf_function=True),
        observers=[rb_observer],
        max_steps=collect_steps_per_iteration,
    )

    for _ in range(num_iterations):

        time_step, _ = collect_driver.run(time_step)


        experience, unused_info = next(iterator)
        experience = (tf.cast(experience[0], tf.int64),) + experience[1:]
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()
        print(step)
        if step % log_interval == 0:
            print("step = {0}: loss = {1}".format(step, train_loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print("step = {0}: Average Return = {1}".format(step, avg_return))
            returns.append(avg_return)


        for metric in metrics:
            metric.tf_summaries(step=step, metrics=metrics)


    policy_dir = "/home/karolito/DL/DQN_game/policy"
    checkpoint_dir = "/home/karolito/DL/DQN_game/checkpoint"
    train_checkpointer = common.Checkpointer(
        ckpt_dir=checkpoint_dir,
        max_to_keep=1,
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,
    )

    tf_policy_saver = policy_saver.PolicySaver(agent.policy)
    tf_policy_saver.save(policy_dir)

    iterations = range(0, num_iterations + 1, eval_interval)

    plt.plot(iterations, returns)

    plt.ylabel("Average Return")
    plt.xlabel("Iterations")
    plt.title("Average Return over Iterations")

    plt.ylim(top=250)

    plt.show()


if __name__ == "__main__":
    main()
