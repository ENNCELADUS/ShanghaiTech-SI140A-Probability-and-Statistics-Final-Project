def epsilon_greedy(epsilon, N, theta):  
    Q = np.zeros(num_arms)
    counts = np.zeros(num_arms)
    
    rewards_history = []
    
    for t in range(1, N+1):
        # Choose arm using epsilon-greedy
        if np.random.rand() < epsilon:
            # Exploration: choose a random arm
            arm = np.random.randint(num_arms)
        else:
            # Exploitation: choose the best arm so far
            arm = np.argmax(Q)
        
        # Simulate pulling the chosen arm and get reward
        reward = 1 if np.random.rand() < theta[arm] else 0
        
        # Update counts and estimates
        counts[arm] += 1
        Q[arm] = Q[arm] + (1/counts[arm])*(reward - Q[arm])
        
        rewards_history.append(reward)
    
    return rewards_history