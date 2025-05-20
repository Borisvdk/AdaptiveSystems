import numpy as np
from maze import Actions


class Policy:
    """Basisklasse voor alle policies"""

    def __init__(self, maze):
        self.maze = maze

    def select_action(self, position):
        """
        Kiest een actie voor de gegeven positie.

        Args:
            position: Huidige positie (rij, kolom)

        Returns:
            Actions: Geselecteerde actie
        """
        raise NotImplementedError("Implementeer deze methode in afgeleide klassen")


class RandomPolicy(Policy):
    """Policy die willekeurige acties selecteert"""

    def select_action(self, position):
        """
        Kiest een willekeurige actie.

        Args:
            position: Huidige positie (rij, kolom)

        Returns:
            Actions: Willekeurige actie
        """
        return np.random.choice(self.maze.actions)


class OptimalPolicy(Policy):
    """Policy die optimale acties selecteert op basis van een policy dictionary"""

    def __init__(self, maze, policy_dict):
        """
        Initialiseert een optimale policy.

        Args:
            maze: Maze instantie
            policy_dict: Dictionary met key=positie, value=beste actie
        """
        super().__init__(maze)
        self.policy_dict = policy_dict

    def select_action(self, position):
        """
        Kiest de beste actie voor de gegeven positie volgens de policy dictionary.

        Args:
            position: Huidige positie (rij, kolom)

        Returns:
            Actions: Beste actie of willekeurige actie als positie niet in policy_dict zit
        """
        if position in self.policy_dict and self.policy_dict[position] is not None:
            return self.policy_dict[position]
        # Fallback naar willekeurige actie als positie niet in policy zit
        return np.random.choice(self.maze.actions)