#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 06:07:21 2025

@author: human
"""

# Import libraries
import pandas as pd
import time
import numpy as np
import warnings
from IPython.display import display, HTML
import yfinance as yf
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
warnings.filterwarnings("ignore")

# Create the environment for reinforcement learning
class StockTradingEnv(gym.Env):
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, stock_data, transaction_cost_percent = 0.005):
        
        super(StockTradingEnv, self).__init__()
        
        # Remove any empty DataFrames
        self.stock_data = {ticker: df for ticker, df in stock_data.items() if not df.empty}
        self.tickers = list(self.stock_data.keys())
        
        if not self.tickers:
            
            raise ValueError("All provided stock data is empty")
        
        # Calculate the size of one stock's data
        sample_df = next(iter(self.stock_data.values()))
        self.n_features = len(sample_df.columns)
        
        self.start_date = sample_df.index[0]
        self.end_date = sample_df.index[-1]
        
        # Define action and observation space
        self.action_space = spaces.Box(low = -1, high = 1, shape = (len(self.tickers),), dtype = np.float32)
        
        # Observation space: price data for each stock + balance + shares held + net worth + max net worth + current step
        self.obs_shape = self.n_features * len(self.tickers) + 2 + len(self.tickers) + 2
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape = (self.obs_shape,), dtype = np.float32)
        
        # Initialize account balance
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.shares_held = {ticker: 0 for ticker in self.tickers}
        self.total_shares_sold = {ticker: 0 for ticker in self.tickers}
        self.total_sales_value = {ticker: 0 for ticker in self.tickers}
        
        # Set the current step
        self.current_step = 0
        
        # Calculate the minimum length of data across all stocks
        self.max_steps = max(0, min(len(df) for df in self.stock_data.values()) - 1)

        # Transaction cost
        self.transaction_cost_percent = transaction_cost_percent

    # Functio to reset
    def reset(self, seed = 100, options = None):
        
        super().reset(seed = seed)
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.shares_held = {ticker: 0 for ticker in self.tickers}
        self.total_shares_sold = {ticker: 0 for ticker in self.tickers}
        self.total_sales_value = {ticker: 0 for ticker in self.tickers}
        self.current_step = 0
        
        return self._next_observation(), {}

    # Function to iterate
    def _next_observation(self):
        
        # initialize the frame
        frame = np.zeros(self.obs_shape)
        
        # Add stock data for each ticker
        idx = 0
        
        # Loop through each ticker
        for ticker in self.tickers:
            
            # Get the DataFrame for the current ticker
            df = self.stock_data[ticker]
            
            # If the current step is less than the length of the DataFrame, add the price data for the current step
            if self.current_step < len(df):
                
                frame[idx:idx + self.n_features] = df.iloc[self.current_step].values
                
            # Otherwise, add the last price data available
            elif len(df) > 0:
                
                frame[idx:idx + self.n_features] = df.iloc[-1].values
                
            # Move the index to the next ticker
            idx += self.n_features

        # Add balance, shares held, net worth, max net worth, and current step
        frame[-4 - len(self.tickers)] = self.balance 
        frame[-3 - len(self.tickers):-3] = [self.shares_held[ticker] for ticker in self.tickers] 
        frame[-3] = self.net_worth 
        frame[-2] = self.max_net_worth 
        frame[-1] = self.current_step 
        
        return frame

    # Function to perform actions
    def step(self, actions):
        
        # update the current step
        self.current_step += 1
        
        # check if we have reached the maximum number of steps
        if self.current_step > self.max_steps:
            
            return self._next_observation(), 0, True, False, {}
        
        current_prices = {}
        
        # Loop through each ticker and perform the action
        for i, ticker in enumerate(self.tickers):
            
            # Get the current price of the stock
            current_prices[ticker] = self.stock_data[ticker].iloc[self.current_step]['price']
            
            # get the action for the current ticker
            action = actions[i]

            # Buy
            if action > 0:
                
                # Calculate the number of shares to buy
                shares_to_buy = int(self.balance * action / current_prices[ticker])
                
                # Calculate the cost of the shares
                cost = shares_to_buy * current_prices[ticker]
                
                # Transaction cost
                transaction_cost = cost * self.transaction_cost_percent
                
                # Update the balance and shares held
                self.balance -= (cost + transaction_cost)
                
                # Update the total shares sold
                self.shares_held[ticker] += shares_to_buy

            # Sell
            elif action < 0:
                
                # Calculate the number of shares to sell
                shares_to_sell = int(self.shares_held[ticker] * abs(action))
                
                # Calculate the sale value
                sale = shares_to_sell * current_prices[ticker]
                
                # Transaction cost
                transaction_cost = sale * self.transaction_cost_percent
                
                # Update the balance and shares held
                self.balance += (sale - transaction_cost)
                
                # Update the total shares sold
                self.shares_held[ticker] -= shares_to_sell
                
                # Update the shares sold
                self.total_shares_sold[ticker] += shares_to_sell
                
                # Update the total sales value
                self.total_sales_value[ticker] += sale
        
        # Calculate the net worth
        self.net_worth = self.balance + sum(self.shares_held[ticker] * current_prices[ticker] for ticker in self.tickers)
        
        # Update the max net worth
        self.max_net_worth = max(self.net_worth, self.max_net_worth)
        
        # Calculate the reward
        reward = self.net_worth - self.initial_balance
        
        # Check if the episode is done
        done = self.net_worth <= 0 or self.current_step >= self.max_steps
        
        obs = self._next_observation()
        
        return obs, reward, done, False, {}

    # Function to visualize
    def render(self, mode = 'human'):
        
        # Print the current step, balance, shares held, net worth, and profit
        profit = self.net_worth - self.initial_balance
        
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance:.2f}')
        
        for ticker in self.tickers:
            
            print(f'{ticker} Shares held: {self.shares_held[ticker]}')
            
        print(f'Net worth: {self.net_worth:.2f}')
        print(f'Profit: {profit:.2f}')

    def close(self):
        
        pass

# Function to perform updates
def update_stock_data(self, new_stock_data, transaction_cost_percent = None):
    
    """
    Update the environment with new stock data.

    Parameters:
    new_stock_data (dict): Dictionary containing new stock data,
                           with keys as stock tickers and values as DataFrames.
    """
    
    # Remove empty DataFrames
    self.stock_data = {ticker: df for ticker, df in new_stock_data.items() if not df.empty}
    self.tickers = list(self.stock_data.keys())

    if not self.tickers:
        
        raise ValueError("All new stock data are empty")

    # Update the number of features if needed
    sample_df = next(iter(self.stock_data.values()))
    self.n_features = len(sample_df.columns)

    # Update observation space
    self.obs_shape = self.n_features * len(self.tickers) + 2 + len(self.tickers) + 2
    self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape = (self.obs_shape,), dtype = np.float32)

    # Update maximum steps
    self.max_steps = max(0, min(len(df) for df in self.stock_data.values()) - 1)

    # Update transaction cost if provided
    if transaction_cost_percent is not None:
        
        self.transaction_cost_percent = transaction_cost_percent

    # Reset the environment
    self.reset()

    print(f"The environment has been updated with {len(self.tickers)} new stocks.")
    
# Define PPO Agent
class PPOAgent:
    
    def __init__(self, env, total_timesteps):
        
        self.model = PPO("MlpPolicy", env, device = 'mps', verbose = 1)
        self.model.learn(total_timesteps = total_timesteps)
    
    def predict(self, obs):
        
        action, _ = self.model.predict(obs)
        
        return action
    
# Define A2C Agent
class A2CAgent:
    
    def __init__(self, env, total_timesteps):
        
        self.model = A2C("MlpPolicy", env, device = 'mps', verbose = 1)
        self.model.learn(total_timesteps = total_timesteps)
    
    def predict(self, obs):
        
        action, _ = self.model.predict(obs)
        return action
    
# Define DDPG Agent
class DDPGAgent:
    
    def __init__(self, env, total_timesteps):
        
        self.model = DDPG("MlpPolicy", env, device = 'mps', verbose = 1)
        self.model.learn(total_timesteps = total_timesteps)
    
    def predict(self, obs):
        
        action, _ = self.model.predict(obs)
        return action
     
# Define SAC Agent
class SACAgent:
    
    def __init__(self, env, total_timesteps):
        
        self.model = SAC("MlpPolicy", env, device = 'mps', verbose = 1)
        self.model.learn(total_timesteps = total_timesteps)
    
    def predict(self, obs):
        
        action, _ = self.model.predict(obs)
        return action
    
# Define TD3 Agent
class TD3Agent:
    
    def __init__(self, env, total_timesteps):
        
        self.model = TD3("MlpPolicy", env, device = 'mps', verbose = 1)
        self.model.learn(total_timesteps = total_timesteps)
    
    def predict(self, obs):
        
        action, _ = self.model.predict(obs)
        
        return action

# Define Ensemble Agent
class EnsembleAgent:
    
    def __init__(self, ppo_model, a2c_model, ddpg_model, sac_model, td3_model): #  
        
        self.ppo_model = ppo_model
        self.a2c_model = a2c_model
        self.ddpg_model = ddpg_model
        self.sac_model = sac_model
        self.td3_model = td3_model
    
    def predict(self, obs):
        
        ppo_action, _ = self.ppo_model.predict(obs)
        a2c_action, _ = self.a2c_model.predict(obs)
        ddpg_action, _ = self.ddpg_model.predict(obs)
        sac_action, _ = self.sac_model.predict(obs)
        td3_action, _ = self.td3_model.predict(obs)
        
        # Average the actions
        ensemble_action = np.mean([ppo_action, a2c_action, ddpg_action, sac_action, td3_action], axis = 0) #ppo_action, a2c_action, 
        
        return ensemble_action
    
    def make_recommendation(self, action):
    
        recommendations = []
    
        for a in action:
        
            if a > .5:
            
                recommendations.append('buy')
        
            elif a < -.5:
            
                recommendations.append('sell')
        
            else:
            
                recommendations.append('hold')
    
        return recommendations

# Function to add technical indicators as features to price data
def add_technical_indicators(df):

    df = df.copy()

    # Calculate EMA 12 and 26 for MACD
    df.loc[:, 'EMA12'] = df['Close'].ewm(span = 12, adjust = False).mean()
    df.loc[:, 'EMA26'] = df['Close'].ewm(span = 26, adjust = False).mean()
    df.loc[:, 'MACD'] = df['EMA12'] - df['EMA26']
    df.loc[:, 'Signal'] = df['MACD'].ewm(span = 9, adjust = False).mean()

    # Calculate short and long rolling average crossovers
    df['crossover_signal'] = 0.0

    # Calculate sma
    df['short'] = df['price'].rolling(window = 30, min_periods = 1, center = False).mean()

    # Calculate lma
    df['long'] = df['price'].rolling(window = 90, min_periods = 1, center = False).mean()

    # Create signals
    df['crossover_signal'][30:] = np.where(df['short'][30:] > df['long'][30:], 1.0, 0.0)

    # Generate trading orders
    df['crossover_positions'] = df['crossover_signal'].diff()

    # Calculate RSI 14
    rsi_14_mode = True
    delta = df['Close'].diff()
    
    if rsi_14_mode:
        
        gain = (delta.where(delta > 0, 0)).rolling(window = 14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window = 14).mean()
        rs = gain / loss
    
    else:
    
        up = delta.where(delta > 0, 0)
        down = -delta.where(delta < 0, 0)
        rs = up.rolling(window = 14).mean() / down.rolling(window = 14).mean()
    
    df.loc[:, 'RSI'] = 100 - (100 / (1 + rs))

    # Calculate CCI 20
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(window=20).mean()
    mean_dev = tp.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
    df.loc[:, 'CCI'] = (tp - sma_tp) / (0.015 * mean_dev)

    # Calculate ADX 14
    high_diff = df['High'].diff()
    low_diff = df['Low'].diff()
    
    df.loc[:, '+DM'] = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    df.loc[:, '-DM'] = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    tr = pd.concat([df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift(1)), np.abs(df['Low'] - df['Close'].shift(1))], axis=1).max(axis=1)
    atr = tr.ewm(span = 14, adjust = False).mean()
    df.loc[:, '+DI'] = 100 * (df['+DM'].ewm(span = 14, adjust = False).mean() / atr)
    df.loc[:, '-DI'] = 100 * (df['-DM'].ewm(span = 14, adjust = False).mean() / atr)
    dx = 100 * np.abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    df.loc[:, 'ADX'] = dx.ewm(span = 14, adjust = False).mean()

    # Drop NaN values
    df.dropna(inplace = True)

    # Keep only the required columns
    df = df[['price', 'Open', 'Close', 'Low', 'High', 'short', 'long', 'crossover_signal', 'crossover_positions', 'Volume', 'MACD', 'Signal', 'RSI', 'CCI', 'ADX']]
    df.columns = [i.lower() for i in df.columns]
    
    return df

# Get historical data from Yahoo Finance and save it to dictionary
def fetch_stock_data(tickers):
    
    stock_data = {}
    
    for ticker in tickers:
        
        data = yf.download(ticker)
        data.columns = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
        data = data.rename({'Adj Close': 'price'}, axis = 1).reset_index()
        data = data.set_index('Date', drop = True)
        stock_data[ticker] = data
        
    return stock_data
    
# Function to create the environment and train the agents
def create_env_and_train_agents(data, total_timesteps):
    
    # Create the environment using DummyVecEnv with training data
    env = DummyVecEnv([lambda: StockTradingEnv(data)])

    # Train PPO Agent
    ppo_agent = PPOAgent(env, total_timesteps)

    # Train A2C Agent
    a2c_agent = A2CAgent(env, total_timesteps)

    # Train DDPG Agent
    ddpg_agent = DDPGAgent(env, total_timesteps)

    # Train SAC Agent
    sac_agent = SACAgent(env, total_timesteps)

    # Train TD3 Agent
    td3_agent = TD3Agent(env, total_timesteps)

    # Train the ensemble agent
    ensemble_agent = EnsembleAgent(ppo_agent.model, a2c_agent.model, ddpg_agent.model, sac_agent.model, td3_agent.model)

    return env, ppo_agent, a2c_agent, ddpg_agent, sac_agent, td3_agent, ensemble_agent

def test_agent(env, agent, stock_data, n_tests, visualize = False):
    
    """
    Test a single agent and track performance metrics, with an option to visualize the results.

    Parameters:
    - env: The trading environment.
    - agent: The agent to be tested.
    - stock_data: Data for the stocks in the environment.
    - n_tests: Number of tests to run (default: 1000).
    - visualize: Boolean flag to enable or disable visualization (default: False).

    Returns:
    - A dictionary containing steps, balances, net worths, and shares held.
    """
    # Initialize metrics tracking
    metrics = {'steps': [],
               'balances': [],
               'net_worths': [],
               'shares_held': {ticker: [] for ticker in stock_data.keys()}}

    # Reset the environment before starting the tests
    obs = env.reset()

    for i in range(n_tests):
        
        metrics['steps'].append(i)
        action = agent.predict(obs)
        obs, rewards, dones, infos = env.step(action)
        
        if visualize:
            
            env.render()

        # Track metrics
        metrics['balances'].append(env.get_attr('balance')[0])
        metrics['net_worths'].append(env.get_attr('net_worth')[0])
        env_shares_held = env.get_attr('shares_held')[0]

        # Update shares held for each ticker
        for ticker in stock_data.keys():
            
            if ticker in env_shares_held:
                
                metrics['shares_held'][ticker].append(env_shares_held[ticker])
                
            else:
                
                metrics['shares_held'][ticker].append(0) 

        if dones:
            
            obs = env.reset()
            
    return metrics

# Function to visualize portfolio changes
def visualize_portfolio(steps, balances, net_worths, shares_held, tickers, show_balance = True, show_net_worth = True, show_shares_held = True):
    
    fig, axs = plt.subplots(3, figsize = (24, 8))

    # Plot the balance
    if show_balance:
        
        axs[0].plot(steps, balances, label = 'Balance')
        axs[0].set_title('Balance Over Time')
        axs[0].set_xlabel('Steps')
        axs[0].set_ylabel('Balance')
        axs[0].legend()

    # Plot the net worth
    if show_net_worth:
        
        axs[1].plot(steps, net_worths, label = 'Net Worth', color = 'orange')
        axs[1].set_title('Net Worth Over Time')
        axs[1].set_xlabel('Steps')
        axs[1].set_ylabel('Net Worth')
        axs[1].legend()

    # Plot the shares held
    if show_shares_held:
        
        for ticker in tickers:
            
            axs[2].plot(steps, shares_held[ticker], label=f'Shares Held: {ticker}')
            
        axs[2].set_title('Shares Held Over Time')
        axs[2].set_xlabel('Steps')
        axs[2].set_ylabel('Shares Held')
        axs[2].legend()

    plt.tight_layout()
    plt.show()

# function to visualize the portfolio net worth
def visualize_portfolio_net_worth(steps, net_worths):
    
    plt.figure(figsize = (24, 8))
    plt.plot(steps, net_worths, label = 'Net Worth', color = 'orange')
    plt.title('Net Worth Over Time')
    plt.xlabel('Steps')
    plt.ylabel('Net Worth')
    plt.legend()
    plt.show()

# function to visualize the multiple portfolio net worths ( same chart )
def visualize_multiple_portfolio_net_worth(steps, net_worths_list, labels):
    
    plt.figure(figsize = (24, 8))
    
    for i, net_worths in enumerate(net_worths_list):
        
        plt.plot(steps, net_worths, label = labels[i])
        
    plt.title('Net Worth Over Time')
    plt.xlabel('Steps')
    plt.ylabel('Net Worth')
    plt.legend()
    plt.show()
    
# Function to compare agents and visualize
def test_and_visualize_agents(env, agents, training_data, n_tests):
    
    metrics = {}
    
    for agent_name, agent in agents.items():
        
        print(f"Testing {agent_name}...")
        metrics[agent_name] = test_agent(env, agent, training_data, n_tests = n_tests, visualize = True)
        print(f"Done testing {agent_name}!")
    
    print('-' * 50)
    print('All agents tested!')
    print('-' * 50)
    
    # Extract net worths for visualization
    net_worths = [metrics[agent_name]['net_worths'] for agent_name in agents.keys()]
    steps = next(iter(metrics.values()))['steps']  
    
    # Visualize the performance metrics of multiple agents
    visualize_multiple_portfolio_net_worth(steps, net_worths, list(agents.keys()))
    
    return metrics


