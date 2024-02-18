import random
from tf_agents.environments import py_environment

class Game(py_environment.PyEnvironment):
    def __init__(self, target_score) -> None:
        self._target_score = target_score
        self._current_score_1 = 0
        self._current_score_2 = 0
        self._turn = 0 if random.randint(0, 1) <= 0.5 else 1
        self._action_space = ["roll", "end_round"]
        self._latest_roll = None
        self._current_iters = 0

    def __str__(self) -> str:
        target_score = f"Target score: {self.get_target_score()}\n"
        player_1_score = f"Player 1 score: {self.get_player_1_score()}\n"
        player_2_score = f"Player 1 score: {self.get_player_2_score()}\n"
        turn = f"Turn: {self.get_turn()}\n"
        current_iters = f"Current iters: {self._current_iters}\n"
        last_roll = f"Last roll: {self.get_roll()}\n"
        return target_score + player_1_score + player_2_score + turn + current_iters + last_roll

    def get_winner(self):
        if self.is_game_over():
            return 0 if self.get_player_1_score() > self.get_player_2_score() else 1

    def get_state(self):
        return self.get_player_1_score(), self.get_player_2_score(), self.get_target_score(), self.get_roll()

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

    def convert_action(self, int_action):
        if int_action == 0:
            return "roll"
        elif int_action == 1:
            return "end_round"

    def step(self, action, turn):
        action_next = self.convert_action(action)
        action = "roll"
        assert action in self.get_action_space(), "Invalid action"


        if action == "roll":
            roll_result = self.roll()
            if roll_result != 1:
                action = action_next
                if not turn:
                    self._current_score_1 += roll_result
                else:
                    self._current_score_2 += roll_result
            else:
                self._turn = 1 - self.get_turn()

        elif action == "end_turn":
            self._turn = 1 - self.get_turn()

        done = self.is_game_over()
        reward = self.calculate_reward()
        self._current_iters += 1
        return self.get_state(), reward, done, {}

    def is_game_over(self):
        if self.get_player_1_score() >= self.get_target_score() or self.get_player_2_score() >= self.get_target_score():
            return True
        return False

    def calculate_reward(self):
        pass

    def reset(self):
        self._current_score_1 = 0
        self._current_score_2 = 0
        self._turn = 0 if random.randint(0, 1) <= 0.5 else 1
        self._action_space = ["roll", "end_round"]
        self._latest_roll = None
        self._current_iters = 0

def main():
    game = Game(target_score=20)
    player_1 = 0
    player_2 = 0
    for _ in range(1_00):
        while game.is_game_over() == False:
            game.step(action=0 if random.randint(0, 1) >= 0.5 else 1, turn=game.get_turn())
        print(game)
        if not game.get_winner():
            player_1 += 1
        else:
            player_2 += 1
        game.reset()
    print(player_1 / 1_000)


if __name__ == "__main__":
    main()