# Subclass the sb3 DQN to allow logging eval auc for hparam tuning

from stable_baselines3 import DQN
from stable_baselines3.common.utils import polyak_update


class CustomDQN(DQN):
    def __init__(self, *args, eval_interval=500, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_auc = 0
        self.eval_interval = eval_interval
        self.eval_env = self.env

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        self._n_calls += 1
        if self._n_calls % self.target_update_interval == 0:
            polyak_update(self.q_net.parameters(),
                          self.q_net_target.parameters(), self.tau)
            # Copy running stats, see GH issue #996
            polyak_update(self.batch_norm_stats,
                          self.batch_norm_stats_target, 1.0)

        self.exploration_rate = self.exploration_schedule(
            self._current_progress_remaining)
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

        # evaluate the agent and log it if step % log_interval == 0:
        if self._n_calls % self.eval_interval == 0:
            eval_rwd = self.evaluate_agent()
            self.eval_auc += eval_rwd
            self.logger.record("Eval. reward:", eval_rwd)
            self.logger.record("eval_auc", self.eval_auc)

    def evaluate_agent(self, n_episodes=1):
        # run the current policy and return the average reward
        avg_reward = 0.
        # Wrap a timelimit:
        # self.eval_env = TimeLimit(self.eval_env, max_episode_steps=500)
        for ep in range(n_episodes):
            state = self.eval_env.reset()
            done = False
            while not done:
                action = self.predict(state, deterministic=True)[0]

                next_state, reward, done, _ = self.eval_env.step(action)

                avg_reward += reward
                state = next_state
                # print(reward)
        avg_reward /= n_episodes
        self.eval_env.close()
        return avg_reward
