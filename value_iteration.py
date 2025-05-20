def value_iteration(maze, gamma=1.0, theta=0.01):
    """
    Voert value iteration uit op de gegeven maze.

    Args:
        maze: De maze omgeving
        gamma: Discount factor
        theta: Convergentie threshold

    Returns:
        tuple: (V: value function dictionary, policy: optimal policy dictionary)
    """
    # Initialiseer alle values op 0, inclusief terminal states
    V = {}
    for i in range(maze.height):
        for j in range(maze.width):
            position = (i, j)
            V[position] = 0

    # Value iteration
    iteration = 0
    while True:
        iteration += 1
        delta = 0
        for i in range(maze.height):
            for j in range(maze.width):
                position = (i, j)
                state = maze.get_state(position)
                if state.is_terminal:
                    # Terminal states blijven op 0
                    continue

                v = V[position]
                # Bepaal de maximale waarde over alle acties
                max_v = float('-inf')
                for action in maze.actions:
                    next_position, reward, _ = maze.deterministic_step(position, action)
                    max_v = max(max_v, reward + gamma * V[next_position])

                V[position] = max_v
                delta = max(delta, abs(v - V[position]))

        print(f"Iteration {iteration}, Delta: {delta:.6f}")
        if delta < theta:
            break

    # Bepaal optimale policy
    policy = {}
    for i in range(maze.height):
        for j in range(maze.width):
            position = (i, j)
            state = maze.get_state(position)
            if state.is_terminal:
                policy[position] = None
                continue

            best_action = None
            best_value = float('-inf')
            for action in maze.actions:
                next_position, reward, _ = maze.deterministic_step(position, action)
                value = reward + gamma * V[next_position]
                if value > best_value:
                    best_value = value
                    best_action = action

            policy[position] = best_action

    return V, policy


def stochastic_value_iteration(maze, gamma=1.0, theta=0.01):
    """
    Voert value iteration uit op een stochastische maze.

    Args:
        maze: De maze omgeving
        gamma: Discount factor
        theta: Convergentie threshold

    Returns:
        tuple: (V: value function dictionary, policy: optimal policy dictionary)
    """
    # Initialiseer alle values op 0, inclusief terminal states
    V = {}
    for i in range(maze.height):
        for j in range(maze.width):
            position = (i, j)
            V[position] = 0  # Nu initialiseren we ALLE states op 0

    # Value iteration
    iteration = 0
    while True:
        iteration += 1
        delta = 0
        for i in range(maze.height):
            for j in range(maze.width):
                position = (i, j)
                state = maze.get_state(position)
                if state.is_terminal:
                    # Terminal states blijven op 0
                    continue

                v = V[position]
                # Bepaal de maximale waarde over alle acties
                max_v = float('-inf')
                for action in maze.actions:
                    expected_value = 0
                    # Hoofdactie heeft 70% kans
                    next_position, reward, _ = maze.deterministic_step(position, action)
                    expected_value += 0.7 * (reward + gamma * V[next_position])

                    # Andere acties hebben elk 10% kans
                    other_actions = [a for a in maze.actions if a != action]
                    for other_action in other_actions:
                        next_position, reward, _ = maze.deterministic_step(position, other_action)
                        expected_value += (0.3 / len(other_actions)) * (reward + gamma * V[next_position])

                    max_v = max(max_v, expected_value)

                V[position] = max_v
                delta = max(delta, abs(v - V[position]))

        print(f"Stochastic Iteration {iteration}, Delta: {delta:.6f}")
        if delta < theta:
            break

    # Bepaal optimale policy
    policy = {}
    for i in range(maze.height):
        for j in range(maze.width):
            position = (i, j)
            state = maze.get_state(position)
            if state.is_terminal:
                policy[position] = None
                continue

            best_action = None
            best_value = float('-inf')
            for action in maze.actions:
                expected_value = 0
                # Hoofdactie heeft 70% kans
                next_position, reward, _ = maze.deterministic_step(position, action)
                expected_value += 0.7 * (reward + gamma * V[next_position])

                # Andere acties hebben elk 10% kans
                other_actions = [a for a in maze.actions if a != action]
                for other_action in other_actions:
                    next_position, reward, _ = maze.deterministic_step(position, other_action)
                    expected_value += (0.3 / len(other_actions)) * (reward + gamma * V[next_position])

                if expected_value > best_value:
                    best_value = expected_value
                    best_action = action

            policy[position] = best_action

    return V, policy
