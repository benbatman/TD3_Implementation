# Import dependencies 
import os 
import time 
import numpy as np 
import matplotlib.pyplot as plt 
#import pybullet_envs
import gym 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 
from collections import deque
import datetime

class ReplayBuffer(object):

  def __init__(self, max_size=1e6):
    self.storage = [] 
    self.max_size = max_size # Maximum transitions memory can store
    self.ptr = 0

  def add(self, transition):
    """
    Adds transitions to memory
    """
    # if memory is fully populated, add new transition to beginning of memory
    if len(self.storage) == self.max_size:
      self.storage[int(self.ptr)] = transition 
      self.ptr = (self.ptr + 1) % self.max_size
    else: # if our memory is not at its full size
      self.storage.append(transition)

  def sample_batch(self, batch_size):
    """
    Method to sample from the replay buffer 
    """
    # Sample batch_size indexes
    ind = np.random.randint(0, len(self.storage), size=batch_size)
    # Initialize to empty lists
    batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
    for i in ind:
      state, next_state, action, reward, done = self.storage[i]
      batch_states.append(np.array(state, copy=False))
      batch_next_states.append(np.array(next_state, copy=False))
      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))
      batch_dones.append(np.array(done, copy=False))
    return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1,1), np.array(batch_dones).reshape(-1,1)

# Build Actor Model and Actor Target

class Actor(nn.Module):
  def __init__(self, state_dim, action_dim, max_action):
    super(Actor, self).__init__() # activate inheritance
    self.layer_1 = nn.Linear(state_dim, 400)
    self.layer_2 = nn.Linear(400,300)
    self.layer_3 = nn.Linear(300,action_dim)
    self.relu = nn.ReLU()
    self.tanh = nn.Tanh()
    self.max_action = max_action

  def forward(self, x):
    # tanh function outputs in range (-1,1), multiplying by max_action value 
    # give range (-num_action, num_action)
    x = self.relu(self.layer_1(x))
    x = self.relu(self.layer_2(x))
    x = self.max_action * self.tanh(self.layer_3(x))
    #x = self.max_action * self.tanh(self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x))))))
    return x

# Build Neural Networks for Critic Model and Target

class Critic(nn.Module):
  
  def __init__(self, state_dim, action_dim, hidden_layer_1 = 400, 
               hidden_layer_2 = 300):
    super(Critic, self).__init__()
    self.input_dim = state_dim + action_dim
    self.hidden_layer_1 = hidden_layer_1
    self.hidden_layer_2 = hidden_layer_2
    # Defining the first Critic neural network
    self.layer_1 = nn.Linear(self.input_dim, self.hidden_layer_1)
    self.layer_2 = nn.Linear(self.hidden_layer_1, self.hidden_layer_2)
    self.layer_3 = nn.Linear(self.hidden_layer_2, 1) # outputs Q-value
    # Defining the second Critic neural network
    self.layer_4 = nn.Linear(self.input_dim, self.hidden_layer_1)
    self.layer_5 = nn.Linear(self.hidden_layer_1, self.hidden_layer_2)
    self.layer_6 = nn.Linear(self.hidden_layer_2, 1) # outputs Q-value
    self.relu = nn.ReLU()

  # u variable is action space
  def forward(self, x, u):
      xu = torch.cat([x,u], dim=1) # concatenate state space and action space
      # Forward-prop on first Critic neural network
      x1 = self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(xu)))))
      # Forward-prop on the second Critic neural network
      x2 = self.layer_6(self.relu(self.layer_5(self.relu(self.layer_4(xu)))))
      return x1, x2 

  # Method to only get Q1
  def Q1(self, x, u):
      xu = torch.cat([x,u], dim=1)
      x1 = self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(xu)))))
      return x1

# Sample Transitions from Experience Replay Buffer

# Get device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TD3(object):

  def __init__(self, state_dim, action_dim, action_max, batch_size=100):
    self.actor = Actor(state_dim, action_dim, action_max).to(device)
    self.actor_target = Actor(state_dim, action_dim, action_max).to(device)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
    self.critic_target = Critic(state_dim, action_dim).to(device)
    self.critic = Critic(state_dim, action_dim).to(device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
    self.action_max = action_max
    self.action_dim = action_dim
    self.state_dim = state_dim

    # if self.save_folder is not None:
    #   test_env = gym.wrappers.Monitor(test_env)

  def get_sample_action(self, state):
    s = torch.Tensor(state.reshape(1,-1)).to(device) # turn state variable into a torch.Tensor
    # Actor target model takes in a state and returns an action
    return self.actor(s).cpu().detach().numpy().flatten()

  # @staticmethod
  # def loss_fn(pred, target):
  #   return nn.MSELoss(pred, target)

  # Training function with default hyperparameters 
  def train(self, replay_buffer, iterations, batch_size=100, gamma=0.99, tau=0.005, 
            noise_scale=0.2, noise_clip=0.5, policy_freq=2):

    for i in range(iterations):
      # Sample a batch of transitions from the replay buffer (s, s', a, r)
      batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample_batch(batch_size)
      state = torch.Tensor(batch_states).to(device)
      next_state = torch.Tensor(batch_next_states).to(device)
      action = torch.Tensor(batch_actions).to(device)
      reward = torch.Tensor(batch_rewards).to(device)
      done = torch.Tensor(batch_dones).to(device)

      # From the next state (s'), the Actor target plays the next action a'
      # print(next_state.shape) # print out shapes for troubleshooting
      # print(action.shape)
      next_action = self.actor_target(next_state)

      # Add Gaussian noise to next_action (a'), clip values in range supported by environment
      noise = torch.Tensor(noise_scale * np.random.randn(self.action_dim)).to(device)
      noise = noise.clamp(-noise_clip, noise_clip)
      next_action = (next_action + noise).clamp(-self.action_max, self.action_max)

      # Critic target networks take each tuple (s',a') as input and returns two Q values
      target_Q1, target_Q2 = self.critic_target(next_state, next_action)

      # Keep the minimum of the two Q-values
      # Represents the approximated value of the next state
      target_Q = torch.min(target_Q1, target_Q2)

      # Get the final target of the two critic models
      target_Q = reward + ((1-done)*gamma*target_Q).detach()

      # Two Critic models take each couple (s,a) as input and return two Q-values
      Q1, Q2 = self.critic(state, action)

      # Compute the loss coming from two Critic Models and sum losses
      #critic_loss = TD3.loss_fn(pred=Q1, target=target_Q) + TD3.loss_fn(pred=Q2, target=target_Q)
      loss = nn.MSELoss()
      critic_loss = loss(Q1, target_Q) + loss(Q2, target_Q)

      # Backprop critic loss and update the parameters of the two critic models 
      self.critic_optimizer.zero_grad() # set gradient to zero first 
      critic_loss.backward() # backprop loss
      self.critic_optimizer.step() #update parameters via Adam optimizer

      # Every two iterations, update Actor model by performing gradent
      # ascent on the output of the first critic model
      if i % policy_freq == 0:
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the weights of the Actor target by polyak averaging
        for params, target_params in zip(self.actor.parameters(), self.actor_target.parameters()):
          target_params.detach().copy_((tau*params.detach()) + (1-tau)*target_params.detach())

        # Update the weights of the Critic target by polyak averaging
        for params, target_params in zip(self.critic.parameters(), self.critic_target.parameters()):
          target_params.detach().copy_((tau*params.detach()) + (1-tau)*target_params.detach())

  # Make save method
  def save_model(self, filename, directory):
    torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
    torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")

  # Make load method 
  def load_model(self, filename, directory):
    self.actor.load_state_dict(torch.load(f"{directory}/{filename}_actor.pth"))
    self.critic.load_state_dict(torch.load(f"{directory}/{filename}_critic.pth"))
  
############################

test_returns = [] # empty list to store returns while testing the agent
def test_agent(policy, max_episode_length, num_episodes=10):
  """
  Function to test agent

  Args: 
  --------------
  policy : a policy for choosing an environment action

  num_episodes : number of episodes agent will test on

  Returns:
  -----------
  avg_reward : array of average returns after testing agent
  
  Appends each return to a list test_returns

  """
  t0 = datetime.datetime.now()
  n_steps = 0
  done = False 
  for _ in range(num_episodes):
    obs, episode_return, episode_length = test_env.reset()[0], 0, 0
    while not done:
      # Take deterministic actions at test time (noise_scale=0)
      a = policy.get_sample_action(np.array(obs))
      obs, r, terminated, truncated, _ = test_env.step(a)
      done = terminated or truncated
      episode_return += r 
      episode_length += 1 
      n_steps += 1  
    print('Test_return:', episode_return, 'Episode_length:', episode_length)
    test_returns.append(episode_return)
  avg_reward = np.asarray(test_returns).mean()
  print(f"Average Reward over the Evaluation period: {avg_reward}")
  print(f"Duration of testing: {datetime.datetime.now()-t0}")
  return avg_reward

###################################################

def mkdir(directory, name):
  path = os.path.join(directory, name)
  if not os.path.exists(path):
    os.makedirs(path)
  return path 




###################################################
def training_loop(env, policy, replay_buffer, num_train_episodes, test_agent_every, 
                  warm_up_steps, batch_size, max_episode_length, save_file_to, save_models=True, gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5,
                  policy_freq=2, expl_noise=0):
  
  """
  Args:
  ---------
  policy : the algorithm to be trained
  num_train_episodes : max number of episodes agent will play
  test_agent_every : how often agent should test on the test environment
  warm_up_steps : how many steps the agent will take random actions before using the policy

  Returns:
  ---------
  """
  # Main loop 
  returns = [] 
  num_steps = 0
  t0 = datetime.datetime.now()

  # Loop through total number of episodes
  for i_episode in range(num_train_episodes):

    # Reset the env, get initial observation 
    obs, episode_return, episode_length = env.reset()[0], 0, 0
    done = False

    # Start episode
    while not done:
      # Randomly sample actions for warmup step number 
      if num_steps < warm_up_steps:
        a = env.action_space.sample()
      else: # Start using policy generated actions 
        a = policy.get_sample_action(np.array(obs))
        # Option to add noise to the action taken (random Gaussian noise)
        if expl_noise != 0:
          a = (a+np.random.normal(0, expl_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)

      # Keep track of the number of steps done 
      num_steps += 1 
      if num_steps == warm_up_steps:
        print("USING AGENT ACTIONS NOW")

      # Agent performs action in environment
      new_obs, r, terminated, truncated, _ = env.step(a)
      done = terminated or truncated
      episode_return += r
      episode_length += 1

      #d_store = False if episode_length == max_episode_length else d # old step api
      d_store = False if truncated else terminated # new step api

      # Store the new transition into the Experience Replay memory
      replay_buffer.add((obs,new_obs,a,r,d_store))

      obs = new_obs

    # Print info and append resturns 
    print(f"Episode: {i_episode} Return: {episode_return} Episode_length {episode_length}")
    returns.append(episode_return)

    # If we aren't at the beginning of the episode, start the training of the model
    # Happens after done flag == True
    # Update agent after it has gotten more experiences 
    if i_episode != 0:
      policy.train(replay_buffer, episode_length, batch_size, gamma, tau, policy_noise, noise_clip, policy_freq)
    
    # Test the agent 
    if i_episode > 0  and i_episode % test_agent_every == 0:
      test_agent(policy, max_episode_length) # appends return to list test_returns
      policy.save_model(save_file_to, directory='./pytorch_models')
      np.savez(f'td3_results_{i_episode}.npz', train=returns, test=test_returns)

  # Print out total training time
  print(f"Total agent training time: {datetime.now()-t0}")

  if save_models:
    policy.save(save_file_to, directory="./pytorch_models")
    # Save the arrays to uncompresed .npz format
    np.savez('final_td3_results.npz', train=returns, test=test_returns)

  # Plot the returns and test_returns
  plt.plot(returns)
  plt.plot(smooth(np.array(returns)))
  plt.title("Train Returns")
  plt.xlabel('episode number')
  plt.ylabel('returns')
  plt.show()

  plt.plot(test_returns)
  plt.plot(smooth(np.array(test_returns)))
  plt.title("Test Returns")
  plt.xlabel("episode_number")
  plt.ylabel("test_returns")
  plt.show()


# Smoothing function (get moving average)
def smooth(x):
  n = len(x)
  y = np.zeros(n)
  for i in range(n):
    start = max(0, i-99)
    y[i] = float(x[start:(i-1)].sum() / (i - start + 1))
  return y


if __name__ == '__main__':
  import argparse 
  parser = argparse.ArgumentParser()
  parser.add_argument('--env', type=str, default='Pendulum-v1')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--save_folder', type=str, default='td3_monitor')
  parser.add_argument('--num_train_episodes', type=int, default=500000)
  parser.add_argument('--gamma', type=int, default=0.99)
  parser.add_argument('--tau', type=int, default=0.005)
  parser.add_argument('--batch_size', type=int)
  parser.add_argument('--save_models', type=bool, default=True)
  parser.add_argument('--test', type=str, default=False)
  args = parser.parse_args()

  print(f"[INFO]\n------------------")
  print(f"gym version: {gym.__version__}")
  print(f"Envrionment name: {args.env}")
  print(f"Number of training episodes: {args.num_train_episodes}")
  print(f"Seed: {args.seed}")
  
  
  if args.test: # do we want to test our agent
    pass

  else: # train the agent
    env, test_env = gym.make(args.env), gym.make(args.env)

    file_name = f"TD3_{args.env}_{str(args.seed)}"

    if not os.path.exists("./results"):
      os.makedirs("./results")
    if args.save_models and not os.path.exists('./pytorch_models'):
      os.makedirs("./pytorch_models")

    # Get the state and action dimensions along with the maximum value allowed for actions taken
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_max = float(env.action_space.high[0])

    print(f"State dimension space: {state_dim}")
    print(f"Action dimension space: {action_dim}")
    print(f"Action max: {action_max}")

    # Set the global seeds
    torch.manual_seed(self.seed)
    np.random.seed(self.seed)


    if args.batch_size:
      policy = TD3(state_dim, action_dim, action_max, batch_size=args.batch_size)
    else:
      policy = TD3(state_dim, action_dim, action_max)
    replay_buffer = ReplayBuffer()
    num_train_episodes = args.num_train_episodes

    training_loop(env=env, policy=policy, replay_buffer=replay_buffer, num_train_episodes=num_train_episodes, test_agent_every=5e3, 
                  warm_up_steps=10000, batch_size=100, max_episode_length=env._max_episode_steps, save_file_to=file_name)

