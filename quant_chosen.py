#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 07:11:01 2025

@author: human
"""

# Import libraries
import os
os.chdir('/Users/human/Documents/quant')
from quantlibrefactored import *
import random
from datetime import date, timedelta

# Choose a list of stocks
# tickers = ['MMM',
#            'TM',
#            'IBM',
#            'MSFT',
#            'GOOG',
#            'AMZN',
#            'SBUX',
#            'TSLA',
#            'AAPL',
#            'CSCO',
#            'JNJ',
#            'NKE', 
#            'PG',
#            'NVDA',
#            'RELX',
#            'BABA']

tickers = ['MMM', 
           'AXP', 
           'AAPL', 
           'BA', 
           'CAT', 
           'CVX', 
           'CSCO', 
           'KO', 
           'DIS',
           'GS', 
           'HD', 
           'IBM', 
           'INTC', 
           'JNJ', 
           'JPM', 
           'MCD', 
           'MRK', 
           'MSFT', 
           'NKE',
           'PFE', 
           'PG', 
           'TRV', 
           'UNH', 
           'VZ', 
           'V', 
           'WBA', 
           'WMT', 
           'XOM']

# Call the function to get data
stock_data = fetch_stock_data(tickers)
    
stock_data_fts = {}

# Use functions to incorporate new features
for ticker, df in stock_data.items():
    
    stock_data_fts[ticker] = add_technical_indicators(df)
    
# # Split the data into training, validation and test sets
# training_data_time_range = ('2000-01-01', '2024-06-15')
# test_data_time_range = ('2024-06-16', '2025-01-13')

# Split the data into training, validation and test sets
training_data_time_range = ('2000-01-01', '2024-06-15')
test_data_time_range = ('2024-06-16', '2025-01-16')

# Split the data into training, validation and test sets
training_data = {}
test_data = {}

for ticker, df in stock_data_fts.items():
    
    training_data[ticker] = df.loc[training_data_time_range[0]:training_data_time_range[1]]
    test_data[ticker] = df.loc[test_data_time_range[0]:]
    
# Get history lengths of training data for all tickers
all_lengths = []

for k, v in training_data.items():

    all_lengths.append((k, v.shape[0]))

all_lengths = pd.DataFrame(all_lengths, columns = ['ticker', 'training_days'])

# Train
total_timesteps = 6152

#env, ppo_agent, a2c_agent, ddpg_agent, ensemble_agent, sac_agent, td3_agent = create_env_and_train_agents(training_data, total_timesteps)

# Create the environment using DummyVecEnv with test data
test_env = DummyVecEnv([lambda: StockTradingEnv(test_data)])
train_env = DummyVecEnv([lambda: StockTradingEnv(training_data)])

# Train Agents
ppo_agent = PPOAgent(train_env, total_timesteps)
a2c_agent = A2CAgent(train_env, total_timesteps)
ddpg_agent = DDPGAgent(train_env, total_timesteps)
sac_agent = SACAgent(train_env, total_timesteps)
td3_agent = TD3Agent(train_env, total_timesteps)
ensemble_agent = EnsembleAgent(ppo_agent.model, a2c_agent.model, ddpg_agent.model, sac_agent.model, td3_agent.model) 

# Compile agents for testing
agents = {'DDPG Agent': ddpg_agent,
          'Ensemble Agent': ensemble_agent, 
          'SAC Agent': sac_agent,
          'TD3 Agent': td3_agent,
          'PPO Agent': ppo_agent,
          'A2C Agent': a2c_agent}

# Iterate the agents and perform stepwise calculations
aggregate_metrics = {}

for agent, actor in agents.items():

    print(f"Testing {agent}...")
    
    # Initialize metrics tracking
    metrics = {'steps': [],
               'balances': [],
               'net_worths': [],
               'shares_held': {ticker: [] for ticker in stock_data.keys()}}

    # Reset the environment before starting the tests
    obs = test_env.reset()

    for i in range(len(test_data['MMM'].index) - 3):
        
        metrics['steps'].append(test_data['MMM'].index[i])
        action = actor.predict(obs)
        obs, rewards, dones, infos = test_env.step(action)
        
        # Track metrics
        metrics['balances'].append(test_env.get_attr('balance')[0])
        metrics['net_worths'].append(test_env.get_attr('net_worth')[0])
        env_shares_held = test_env.get_attr('shares_held')[0]

        # Update shares held for each ticker
        for ticker in stock_data.keys():
            
            if ticker in env_shares_held:
                
                metrics['shares_held'][ticker].append(env_shares_held[ticker])
                
            else:
                
                metrics['shares_held'][ticker].append(0) 
    
    aggregate_metrics[agent] = metrics
    
    print(f"Done testing {agent}!\n-----------------------------")

# Extract net worths for visualization
net_worths = [aggregate_metrics[agent_name]['net_worths'] for agent_name in agents.keys()]
labels = [i for i in agents.keys()]
steps = [aggregate_metrics[agent_name]['steps'] for agent_name in agents.keys()]

# Visualize
plt.figure(figsize = (24, 8))
             
for i, (nw, s) in enumerate(zip(net_worths, steps)):
    
    plt.plot(s, nw, label = labels[i])
                 
plt.title('Portfolio Net Worth Over Time for Trading Agents')
plt.xlabel('Date')
plt.ylabel('Net Worth')
plt.legend()
plt.show()

# Check final balances
for k, v in aggregate_metrics.items():

    print(f"Statistics for {k}: {str(v['steps'][0])[0:10]} :: {str(v['steps'][-1])[0:10]}")
    print(f"Net worth-                {round(v['net_worths'][-1], 2)}")
    print(f"Aggregate Return-         {round(v['net_worths'][-1] / 10000, 2)}")
