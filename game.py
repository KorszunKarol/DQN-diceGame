import random
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
import numpy as np


class Game(py_environment.PyEnvironment):
    def __init__(self, target_score) -> None:
        self._target_score = target_score
        self._current_score_1 = 0
        self._current_score_2 = 0
        self._turn = 0 if random.randint(0, 1) <= 0.5 else 1
        self._previous_turn = 1 - self._turn
        self._action_space = ["roll", "end_round"]
        self._latest_roll = None
        self._current_iters = 0
        self._sum = 0
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1, name="action"
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(np.array(self.get_state()).shape),
            dtype=np.int32,
            minimum=0,
            name="observation",
        )
        self._state = [
            self.get_player_1_score(),
            self.get_player_2_score(),
            self.get_target_score(),
            self.get_sum(),
        ]
        self._episode_ended = False

    def get_sum(self):
        return self._sum

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = [0, 0, self.get_target_score(), 0]
        self._current_score_1 = 0
        self._current_score_2 = 0
        self._episode_ended = False
        return ts.restart(np.array(self.get_state(), dtype=np.int32))

    def __str__(self) -> str:
        sum_1 = self.get_sum() if not self.get_turn() else 0
        sum_2 = self.get_sum() if self.get_turn() else 0
        target_score = f"Target score: {self.get_target_score()}\n"
        player_1_score = f"Player 1 score: {self.get_player_1_score() + sum_1}\n"
        player_2_score = f"Player 2 score: {self.get_player_2_score() + sum_2}\n"
        turn = f"Turn: {'Player 2' if self.get_turn() else 'Player 1'}\n"
        current_iters = f"Current iters: {self._current_iters}\n"
        last_roll = f"Last roll: {self.get_roll()}\n"
        return (
            target_score
            + player_1_score
            + player_2_score
            + turn
            + current_iters
            + last_roll
        )

    def get_winner(self):
        if self.is_game_over():
            return 0 if self.get_player_1_score() > self.get_player_2_score() else 1

    def get_state(self):
        return [
            self.get_player_1_score(),
            self.get_player_2_score(),
            self.get_target_score(),
            self.get_sum(),
        ]

    def get_target_score(self):
        return self._target_score

    def get_player_1_score(self):
        return self._current_score_1

    def get_player_2_score(self):
        return self._current_score_2

    def get_turn(self):
        return self._turn

    def roll(self):
        roll = random.randint(1, 6)
        self._latest_roll = roll
        return roll

    def get_roll(self):
        return self._latest_roll

    def get_action_space(self):
        return self._action_space

    def episode_ended(self):
        return self._episode_ended

    def convert_action(self, int_action):
        if int_action == 0:
            return "roll"
        elif int_action == 1:
            return "end_round"

    def add_points(self):
        if not self.get_turn():
            self._current_score_1 += self.get_sum()
        else:
            self._current_score_2 += self.get_sum()

    def roll_once(self):
        roll_result = self.roll()
        if roll_result != 1:
            if not self.get_turn():
                self._current_score_1 += roll_result
            else:
                self._current_score_2 += roll_result
        else:
            self._turn = 1 - self.get_turn()

    def _step(self, action):

        if self.episode_ended():
            return self.reset()
        if self._previous_turn != self.get_turn() or action == 2:
            self._previous_turn = self.get_turn()
            self.roll_once()
            if action == 2:
                if self.is_game_over():
                    self._episode_ended = True
                    reward = self.calculate_reward()
                    return ts.termination(
                        np.array(self.get_state(), dtype=np.int32), reward
                    )
                return ts.transition(
                    np.array(self.get_state(), dtype=np.int32), reward=0.0, discount=1.0
                )

        action = self.convert_action(action)

        self._previous_turn = self.get_turn()

        assert action in self.get_action_space(), "Invalid action"

        if action == "roll":
            roll_result = self.roll()

            if roll_result != 1:
                self._sum += roll_result
            else:
                self._sum = 0
                self._turn = 1 - self.get_turn()

        elif action == "end_round":
            self.add_points()
            self._sum = 0
            self._turn = 1 - self.get_turn()

        self._current_iters += 1

        if self.is_game_over():
            self.reset_scores()
            self._episode_ended = True
            reward = self.calculate_reward()
            return ts.termination(np.array(self.get_state(), dtype=np.int32), reward)
        else:
            return ts.transition(
                np.array(self.get_state(), dtype=np.int32),
                reward=self.calculate_reward(),
                discount=1.0,
            )

    def is_game_over(self):
        if (
            self.get_player_1_score() >= self.get_target_score()
            or self.get_player_2_score() >= self.get_target_score()
        ):
            return True
        return False

    def reset_scores(self):
        self._current_score_1 = 0
        self._current_score_2 = 0
        self._sum = 0
        self._turn = 0 if random.randint(0, 1) <= 0.5 else 1

    def calculate_reward(self):
        score_1, score_2, target, _ = self.get_state()
        reward = 2 * (score_1 - score_2)
        if self.is_game_over():
            if score_1 > score_2:
                reward += reward * 2
            elif score_2 > score_1:
                reward -= reward * 2
        return reward

    def play_against_policy(self, policy):
        self._reset()
        while not self.is_game_over():
            roll_result = self.roll()
            if roll_result == 1:
                self._turn = 1 - self.get_turn()
                continue
            else:
                self._current_score_1 += roll_result if not self.get_turn() else 0
                self._current_score_2 += roll_result if self.get_turn() else 0
            if not self.get_turn():
                action = self.convert_action(
                    int(input("Make a decision: (0 - roll again 1 - end round)"))
                )
                if action == "roll":
                    roll_result = self.roll()
                elif action == "end_round":
                    self._turn = 1 - self.get_turn()
                    continue
            else:
                pass


def main():
    policy = tf.saved_model.load("policy")
    print(type(policy))
    environment = Game(target_score=50)

    winner_1 = 0
    winner_2 = 0
    env = tf_py_environment.TFPyEnvironment(environment, isolation=True)
    for _ in range(50):

        time_step = env.reset()
        episode_return = 0.0
        while not time_step.is_last():
            if (
                max(environment.get_player_1_score(), environment.get_player_2_score())
                > environment.get_target_score()
            ):
                break
            print(environment.get_state())

            if environment.get_turn():

                action = random.randint(0, 1)
                time_step = env.step(action)
            else:
                action_step = policy.action(time_step)

                time_step = env.step(action_step.action)
                episode_return += time_step.reward

        if environment.get_player_1_score() > environment.get_player_2_score():
            winner_1 += 1
        else:
            winner_2 += 1

    print(winner_1 / (winner_1 + winner_2))


if __name__ == "__main__":
    main()
