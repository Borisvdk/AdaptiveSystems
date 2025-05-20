import numpy as np
import matplotlib.pyplot as plt
from maze import Actions


def visualize_maze(maze, values=None, policy=None, title="Maze Environment"):
    """
    Visualiseert de maze, optioneel met value function en policy.

    Args:
        maze: Maze instantie
        values: Dictionary van values voor elke positie (optioneel)
        policy: Dictionary van acties voor elke positie (optioneel)
        title: Titel voor de plot
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Tekenen van het grid
    for i in range(maze.height + 1):
        ax.axhline(i, color='black')
    for j in range(maze.width + 1):
        ax.axvline(j, color='black')

    # Vullen van de cellen met rewards
    for i in range(maze.height):
        for j in range(maze.width):
            position = (i, j)
            state = maze.get_state(position)
            reward = state.reward

            # Stel de celkleur in op basis van de reward
            if reward > 0:
                cell_color = 'lightgreen'
            elif reward < -5:
                cell_color = 'salmon'
            else:
                cell_color = 'white'

            # Teken de cel
            rect = plt.Rectangle((j, maze.height - 1 - i), 1, 1, facecolor=cell_color, alpha=0.5)
            ax.add_patch(rect)

            # Teken reward
            ax.text(j + 0.5, maze.height - 1 - i + 0.2, f"R: {reward}",
                    ha='center', va='center', fontsize=10)

            # Teken value als beschikbaar
            if values and position in values:
                ax.text(j + 0.5, maze.height - 1 - i + 0.5, f"V: {values[position]:.1f}",
                        ha='center', va='center', fontsize=10, color='blue')

            # Teken policy als beschikbaar
            if policy and position in policy and policy[position] is not None:
                action = policy[position]
                # Map Actions enum naar symbolen
                action_symbols = {
                    Actions.LEFT: '←',
                    Actions.UP: '↑',
                    Actions.RIGHT: '→',
                    Actions.DOWN: '↓'
                }
                ax.text(j + 0.5, maze.height - 1 - i + 0.8, action_symbols[action],
                        ha='center', va='center', fontsize=20, color='red')

    # Markeer terminal states
    for position in maze.terminal_positions:
        i, j = position
        plt.text(j + 0.5, maze.height - 1 - i + 0.65, "FINISH",
                 ha='center', va='center', fontsize=10, color='green')

    # Markeer start state
    i, j = maze.start_position
    plt.text(j + 0.5, maze.height - 1 - i + 0.65, "START",
             ha='center', va='center', fontsize=10, color='purple')

    ax.set_xlim(0, maze.width)
    ax.set_ylim(0, maze.height)
    ax.set_xticks(np.arange(0.5, maze.width))
    ax.set_yticks(np.arange(0.5, maze.height))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def visualize_episode(maze, path, title="Agent Path"):
    """
    Visualiseert het pad van een agent door de maze.

    Args:
        maze: Maze instantie
        path: Lijst van posities (rij, kolom) die het pad vormen
        title: Titel voor de plot
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Tekenen van het grid
    for i in range(maze.height + 1):
        ax.axhline(i, color='black')
    for j in range(maze.width + 1):
        ax.axvline(j, color='black')

    # Vullen van de cellen met rewards
    for i in range(maze.height):
        for j in range(maze.width):
            position = (i, j)
            state = maze.get_state(position)
            reward = state.reward

            # Stel de celkleur in op basis van de reward
            if reward > 0:
                cell_color = 'lightgreen'
            elif reward < -5:
                cell_color = 'salmon'
            else:
                cell_color = 'white'

            # Teken de cel
            rect = plt.Rectangle((j, maze.height - 1 - i), 1, 1, facecolor=cell_color, alpha=0.5)
            ax.add_patch(rect)

            # Teken reward
            ax.text(j + 0.5, maze.height - 1 - i + 0.5, f"R: {reward}",
                    ha='center', va='center', fontsize=10)

    # Teken het pad
    path_x = [j + 0.5 for i, j in path]
    path_y = [maze.height - 1 - i + 0.5 for i, j in path]
    ax.plot(path_x, path_y, 'bo-', markersize=10, alpha=0.7)

    # Markeer start en eind van het pad
    ax.plot(path_x[0], path_y[0], 'go', markersize=15)  # Start
    ax.plot(path_x[-1], path_y[-1], 'ro', markersize=15)  # Eind

    # Markeer terminal states
    for position in maze.terminal_positions:
        i, j = position
        plt.text(j + 0.5, maze.height - 1 - i + 0.8, "FINISH",
                 ha='center', va='center', fontsize=10, color='green')

    # Markeer start state
    i, j = maze.start_position
    plt.text(j + 0.5, maze.height - 1 - i + 0.8, "START",
             ha='center', va='center', fontsize=10, color='purple')

    ax.set_xlim(0, maze.width)
    ax.set_ylim(0, maze.height)
    ax.set_xticks(np.arange(0.5, maze.width))
    ax.set_yticks(np.arange(0.5, maze.height))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def compare_policies(maze, deterministic_values, deterministic_policy,
                     stochastic_values, stochastic_policy):
    """
    Vergelijkt deterministische en stochastische policies.

    Args:
        maze: Maze instantie
        deterministic_values: Value function voor deterministische omgeving
        deterministic_policy: Policy voor deterministische omgeving
        stochastic_values: Value function voor stochastische omgeving
        stochastic_policy: Policy voor stochastische omgeving
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Hulpfunctie voor het plotten
    def plot_maze_on_axis(ax, values, policy, title):
        # Tekenen van het grid
        for i in range(maze.height + 1):
            ax.axhline(i, color='black')
        for j in range(maze.width + 1):
            ax.axvline(j, color='black')

        # Vullen van de cellen
        for i in range(maze.height):
            for j in range(maze.width):
                position = (i, j)
                state = maze.get_state(position)
                reward = state.reward

                # Stel de celkleur in op basis van de reward
                if reward > 0:
                    cell_color = 'lightgreen'
                elif reward < -5:
                    cell_color = 'salmon'
                else:
                    cell_color = 'white'

                # Teken de cel
                rect = plt.Rectangle((j, maze.height - 1 - i), 1, 1, facecolor=cell_color, alpha=0.5)
                ax.add_patch(rect)

                # Teken value
                if position in values:
                    ax.text(j + 0.5, maze.height - 1 - i + 0.3, f"V: {values[position]:.1f}",
                            ha='center', va='center', fontsize=9, color='blue')

                # Teken policy
                if position in policy and policy[position] is not None:
                    action = policy[position]
                    # Map Actions enum naar symbolen
                    action_symbols = {
                        Actions.LEFT: '←',
                        Actions.UP: '↑',
                        Actions.RIGHT: '→',
                        Actions.DOWN: '↓'
                    }
                    ax.text(j + 0.5, maze.height - 1 - i + 0.7, action_symbols[action],
                            ha='center', va='center', fontsize=18, color='red')

        # Markeer terminal states
        for position in maze.terminal_positions:
            i, j = position
            ax.text(j + 0.5, maze.height - 1 - i + 0.5, "FINISH",
                    ha='center', va='center', fontsize=9, color='green')

        # Markeer start state
        i, j = maze.start_position
        ax.text(j + 0.5, maze.height - 1 - i + 0.5, "START",
                ha='center', va='center', fontsize=9, color='purple')

        ax.set_xlim(0, maze.width)
        ax.set_ylim(0, maze.height)
        ax.set_xticks(np.arange(0.5, maze.width))
        ax.set_yticks(np.arange(0.5, maze.height))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(title)

    # Plot deterministische policy
    plot_maze_on_axis(ax1, deterministic_values, deterministic_policy, "Deterministic Environment Policy")

    # Plot stochastische policy
    plot_maze_on_axis(ax2, stochastic_values, stochastic_policy, "Stochastic Environment Policy")

    plt.tight_layout()
    plt.show()