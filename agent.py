class Agent:
    """Agent die door de doolhof navigeert"""

    def __init__(self, maze, policy):
        """
        Initialiseert een agent.

        Args:
            maze: Maze instantie
            policy: Policy instantie die acties selecteert
        """
        self.maze = maze
        self.policy = policy
        self.value_function = {}  # Dictionary om value function op te slaan

    def act(self, position, stochastic=False):
        """
        Voert een actie uit op basis van de policy.

        Args:
            position: Huidige positie (rij, kolom)
            stochastic: Of de omgeving stochastisch moet zijn

        Returns:
            tuple: (volgende positie, reward, done, actie)
        """
        action = self.policy.select_action(position)
        next_position, reward, done = self.maze.step(position, action, stochastic)
        return next_position, reward, done, action

    def simulate_episode(self, start_position=None, max_steps=100, stochastic=False):
        """
        Simuleert een volledige episode.

        Args:
            start_position: Beginpositie (standaard: maze.start_position)
            max_steps: Maximum aantal stappen om oneindige lussen te voorkomen
            stochastic: Of de omgeving stochastisch moet zijn

        Returns:
            tuple: (pad [lijst van posities], totale beloning, aantal stappen)
        """
        if start_position is None:
            current_position = self.maze.start_position
        else:
            current_position = start_position

        total_reward = 0
        path = [current_position]
        steps = 0

        while not self.maze.is_terminal(current_position) and steps < max_steps:
            next_position, reward, done, _ = self.act(current_position, stochastic)
            total_reward += reward
            current_position = next_position
            path.append(current_position)
            steps += 1

        return path, total_reward, steps