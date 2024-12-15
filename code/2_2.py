def ucb(c, N, theta):
    num_arms = len(theta)
    # Estimated values for each arm
    Q = np.zeros(num_arms)
    # Count of how many times each arm is pulled
    counts = np.zeros(num_arms)
    
    rewards_history = []
    
    # Initialization: pull each arm once
    for t in range(num_arms):
        arm = t
        reward = 1 if np.random.rand() < theta[arm] else 0
        counts[arm] += 1
        Q[arm] = reward
        rewards_history.append(reward)
    
    # Main loop
    for t in range(num_arms+1, N+1):
        # Compute UCB values
        ucb_values = Q + c * np.sqrt((2*np.log(t)) / counts)
        
        arm = np.argmax(ucb_values)
        
        reward = 1 if np.random.rand() < theta[arm] else 0
        counts[arm] += 1
        Q[arm] += (1/counts[arm])*(reward - Q[arm])
        rewards_history.append(reward)
    
    return np.array(rewards_history)