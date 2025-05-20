import numpy as np
from enum import Enum


class Actions(Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3


class State:
    """Representeert een state in de doolhof"""

    def __init__(self, position, reward, is_terminal):
        """
        Initialiseert een State.

        Args:
            position: Tuple (rij, kolom) die de positie aangeeft
            reward: De reward verbonden aan deze state
            is_terminal: Boolean die aangeeft of dit een terminal state is
        """
        self.position = position
        self.reward = reward
        self.is_terminal = is_terminal

    def __eq__(self, other):
        if isinstance(other, State):
            return self.position == other.position
        return False

    def __hash__(self):
        return hash(self.position)


class Maze:
    def __init__(self):
        # Grid afmetingen
        self.height = 4
        self.width = 4

        # Acties als Enum
        self.actions = [a for a in Actions]

        # Rewards voor elke positie (rij, kolom)
        self.rewards_grid = np.array([
            [-1, -1, -1, 40],  # Bovenste rij
            [-1, -1, -10, -10],  # Tweede rij
            [-1, -1, -1, -1],  # Derde rij
            [10, -2, -1, -1]  # Onderste rij
        ])

        # Terminal positions
        self.terminal_positions = [(0, 3), (3, 0)]

        # Start position
        self.start_position = (3, 2)

        # Verzameling van states (gevraagd in de opdracht)
        self.states = {}
        for i in range(self.height):
            for j in range(self.width):
                position = (i, j)
                reward = self.rewards_grid[i][j]
                is_terminal = position in self.terminal_positions
                self.states[position] = State(position, reward, is_terminal)

    def get_state(self, position):
        """Haalt een state object op basis van positie"""
        return self.states[position]

    def is_terminal(self, position):
        """Controleert of een positie een terminal state is"""
        return self.states[position].is_terminal

    def get_reward(self, position):
        """Haalt reward op voor een bepaalde positie"""
        return self.states[position].reward

    def get_next_position(self, position, action):
        """Bepaalt de volgende positie na uitvoeren van een actie"""
        row, col = position

        # Bepaal de nieuwe positie op basis van actie
        if action == Actions.LEFT:
            col = max(0, col - 1)
        elif action == Actions.UP:
            row = max(0, row - 1)
        elif action == Actions.RIGHT:
            col = min(self.width - 1, col + 1)
        elif action == Actions.DOWN:
            row = min(self.height - 1, row + 1)

        return (row, col)

    def step(self, position, action, stochastic=False):
        """
        Voert een stap in de omgeving uit.

        Args:
            position: Huidige positie (rij, kolom)
            action: Te ondernemen actie (Actions enum)
            stochastic: Of de omgeving stochastisch moet zijn (70% kans op gekozen actie)

        Returns:
            tuple: (volgende positie, reward, done)
        """
        if not stochastic:
            return self.deterministic_step(position, action)

        # Stochastische uitvoering: 70% kans op gekozen actie, 30% verdeeld over andere richtingen
        p = np.zeros(len(self.actions))
        p[action.value] = 0.7  # 70% kans op gekozen actie

        # Verdeel overige 30% over andere acties
        other_actions = [a for a in self.actions if a != action]
        for a in other_actions:
            p[a.value] = 0.3 / len(other_actions)

        # Kies actie volgens de kansverdelingen
        action_values = [a.value for a in self.actions]
        actual_action = self.actions[np.random.choice(action_values, p=p)]

        return self.deterministic_step(position, actual_action)

    def deterministic_step(self, position, action):
        """Deterministische stap in de omgeving"""
        next_position = self.get_next_position(position, action)
        reward = self.get_reward(next_position)
        done = self.is_terminal(next_position)

        return next_position, reward, done
