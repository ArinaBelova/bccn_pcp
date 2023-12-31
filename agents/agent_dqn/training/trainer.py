import gym
import gym_grid

import numpy as np
import torch
import torch.nn as nn

from mlp import MLP
from policies import eps_greedy

from agents.agent_dqn.experience_buffer import ExperienceBuffer

class Trainer:

  DEFAULTS = {
    'loss_function': nn.MSELoss(),
    'learning_rate': 0.001,
    'policy': eps_greedy,
    'exp_replay': False,
    'num_update_steps': 20000,
    'num_episodes': 50,
    'eval_steps': 20,
    'evaluate_at': 500,
    'debug': False,
    'gamma': 0.99
  }

  def __init__(self, **params: dict) -> None:
    # Required params
    self.env_name = params["env_name"]
    self.model = self.create_model()

    # Params that have defaults
    self.loss_function = params.get("loss_function", self.DEFAULTS["loss_function"])
    self.lr = params.get("learning_rate", self.DEFAULTS["learning_rate"])
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

  def the_loop(self, **params):
    exp_replay = params.get("exp_replay", self.DEFAULTS["exp_replay"])
    num_update_steps = params.get("num_update_steps", self.DEFAULTS["num_update_steps"])
    num_episodes = params.get("num_episodes", self.DEFAULTS["num_episodes"])
    policy = params.get("policy", self.DEFAULTS["policy"])
    eval_steps = params.get("eval_steps", self.DEFAULTS["eval_steps"])
    evaluate_at = params.get("evaluate_at", self.DEFAULTS["evaluate_at"])
    gamma = params.get("gamma", self.DEFAULTS["gamma"])
    debug = params.get("debug", self.DEFAULTS["debug"])

    # episodes num = 50/100 and updates = 20000/100000
    #for update_step in range(num_update_steps):
    main_env = self.create_env()
    rew_inter_arr = np.zeros(int(num_update_steps / evaluate_at))
    discounted_rew_arr = np.zeros(int(num_update_steps / evaluate_at))
    counter = 0
    sample_size = np.prod(main_env.observation_space.shape)

    # So epsilon decays only in 80% of the steps and in first 20% of the steps it stays exploratative
    epsilon = lambda count: 1 * (1 - ((count - 0.2 * num_update_steps) / num_update_steps)) if count >= 0.2 * num_update_steps else 1
    #print(f"Do we use experience replay? {exp_replay}")
    if exp_replay:
      exp_buffer = ExperienceBuffer((sample_size,1,1,sample_size,1))
    else:
      exp_buffer = ExperienceBuffer((sample_size,1,1,sample_size,1), batch_size = 1)

    #print(f"We use experience replay buffer with buffer size {exp_buffer.batch_size}")  

    # TODO: num of episodes * num_steps_in_episode = 20000
    while counter < num_update_steps:
      current_state = main_env.reset()
      current_state_tensor = torch.Tensor(current_state.flatten())

      # TODO: num steps in the episode = 50
      # num episodes is not fixed!
      for _ in range(num_episodes):
        # 1. Forward pass on NN to get the Q(t)
        # Not to reset current_state_tensor at every pass of for loop
        # if idx == 0:
        #   current_state_tensor = torch.Tensor(current_state.flatten())
        q_t = self.model(current_state_tensor).detach()

        # 2. Decide which action to choose based on annealing epsilon greedy policy
        # TODO: # sigmoid schedule on the epsilon so we have more exploration. We need to do annealing for 80% of steps
        # Clipping
        chosen_action, _ = policy(q_t, e = epsilon(counter))
        #print(f"Current q value is {current_q_value}")

        # 3. Take action/step and get reward and next state
        next_state, reward , done, _ = main_env.step(chosen_action)

        # 4. Run another forward pass to get all q_values for next_state
        next_state_tensor = torch.Tensor(next_state.flatten())

        # 4(a). Store in experence replay buffer:
        #print(current_state_tensor)
        exp_buffer.add(current_state_tensor, chosen_action, reward, next_state_tensor, done)
      
        # Important previously: current_state_tensor = next_state_tensor
        current_state_tensor = next_state_tensor

        # WIP: Now we ask for a batch and we update the gradient wrt batch
        # first check is whether we have sufficiently full buffer
        if counter >= exp_buffer.batch_size:
          #for _ in range(exp_buffer.DEFAULT_BATCH_SIZE):
          buffer_batch = exp_buffer.get_next()

          q_current_t = self.model(buffer_batch.get_current_states())
          #print(f"Current q shape: {q_current_t.shape}")
          _, current_max_q_value = policy(q_current_t, e = epsilon(counter))
          #print(f"Current q after policy shape: {current_max_q_value.shape}")

          q_next_t = self.model(buffer_batch.get_next_states()) # will be a matrix of Qs (buffer_size x num_actions)
          #print(q_next_t.shape)

          # 5. Get maximum q_value, greedy action
          _, next_max_q_value = policy(q_next_t, e=0)
          #print(next_max_q_value)
          #next_max_q_value, q_next_t

          # 6. Calculate Y_hat
          y_hat = Trainer.calculate_y_hat(buffer_batch.get_rewards(), next_max_q_value, buffer_batch.get_done() * 1, gamma=gamma).detach()

          # 7. Loss function
          loss = self.loss_function(y_hat, current_max_q_value)

          # 8. Backpropagate to learn
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()

        # 9. TODO: Do the evaluation every 500 steps
        # run in the env under greedy policy to get running reward
        if (counter % evaluate_at) == 0:
          with torch.no_grad():
            rew_intermediate, discounted_rew = self.rollout_episode(policy, eval_steps, gamma=gamma)
            rew_inter_arr[int(counter / evaluate_at)] = rew_intermediate
            discounted_rew_arr[int(counter / evaluate_at)] = discounted_rew
            print(f"Intermediate reward at idx {int(counter / evaluate_at)} update is {rew_intermediate}")
            print(f"Discounted reward at idx {int(counter / evaluate_at)} update is {discounted_rew}")
          if debug:
            print("\n======= Debug =========")
            print("num_update_steps: ", num_update_steps)
            print("counter: ", counter)
            print("evaluate_at: ", evaluate_at)
            if counter >= exp_buffer.batch_size:
              print("y_hat shape", y_hat.shape)
              print("current_max_q_value shape", current_max_q_value.shape)
            print("-- [exp_replay]", exp_buffer)
            print("======= End Debug =========")

        # step counter
        counter +=1

        # 11. Check if we're out of updating for good.
        if done or (counter >= num_update_steps):
          break
        
        # 10. Check if we terminate an episode due to one of the termination conditions
        # if done:
        #   break


    return rew_inter_arr, discounted_rew_arr

  @staticmethod
  def calculate_y_hat(reward, max_next_q, done, gamma=0.99):
    return reward.squeeze() + gamma * (torch.ones(done.shape[0]) - done.squeeze()) * max_next_q.detach()
    #return torch.Tensor([reward + gamma * (1 - done) * max_next_q.detach().numpy()])
  
  def rollout_episode(self, policy, eval_steps, eval_episodes = 3, gamma = 0.99):
    self.model.eval()
    rewards = []
    d_rewards = []
    # TODO: should we really do reset of the env?
    # We need a separate testing environment!!!
    test_env = self.create_env()

    for _ in range(eval_episodes):
      reward = 0
      d_reward = 0
      current_state = test_env.reset()

      # Multiple episodes not steps!!! And avg discounted reward over episodes.
      for t in range(eval_steps):
        # use the epsilon greedy policy with a ver small epsilon = 0.01
        q = self.model(torch.Tensor(current_state.flatten()))
        action, _ = policy(q, e = 0.01)
        new_state, rew , done, _ = test_env.step(action)
        reward += rew
        d_reward += rew * pow(gamma, t)
        current_state = torch.Tensor(new_state.flatten())
        if done:
          break
      rewards.append(reward)
      d_rewards.append(d_reward)

    self.model.train()
    return np.mean(rewards), np.mean(d_rewards)

  def create_env(self):
    return gym.make(self.env_name)

  def create_model(self):
    temp_env = self.create_env()
    states_size = np.prod(temp_env.observation_space.shape)
    actions_size = temp_env.action_space.n

    return MLP(states_size, actions_size)
