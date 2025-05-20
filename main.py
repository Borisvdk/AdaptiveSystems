import numpy as np
import matplotlib.pyplot as plt
from maze import Maze, Actions
from policies import RandomPolicy, OptimalPolicy
from agent import Agent
from value_iteration import value_iteration, stochastic_value_iteration
from visualization import visualize_maze, visualize_episode, compare_policies


def main():
    # Seed voor reproduceerbaarheid
    np.random.seed(42)

    # Creëer maze
    maze = Maze()

    print("Maze Rewards:")
    print(maze.rewards_grid)

    # Visualiseer de maze
    visualize_maze(maze, title="Maze Environment with Rewards")

    # A. Random Agent Test
    random_policy = RandomPolicy(maze)
    random_agent = Agent(maze, random_policy)

    # Simuleer een episode met random agent
    path, total_reward, steps = random_agent.simulate_episode()
    print(f"\nRandom Agent Path Length: {steps}")
    print(f"Random Agent Total Reward: {total_reward}")

    # Visualiseer de random agent episode
    visualize_episode(maze, path, title=f"Random Agent Path (Reward: {total_reward})")

    # B. Value Iteration (Deterministic)
    print("\nRunning Value Iteration (Deterministic)...")
    deterministic_values, deterministic_policy = value_iteration(maze, gamma=1.0, theta=0.01)

    # Visualiseer value function en policy
    visualize_maze(maze, deterministic_values, deterministic_policy,
                   title="Deterministic Value Function and Policy")

    # Test optimal agent
    optimal_policy = OptimalPolicy(maze, deterministic_policy)
    optimal_agent = Agent(maze, optimal_policy)

    # Simuleer een episode met optimal agent
    optimal_path, optimal_reward, optimal_steps = optimal_agent.simulate_episode()
    print(f"\nOptimal Agent Path Length: {optimal_steps}")
    print(f"Optimal Agent Total Reward: {optimal_reward}")

    # Visualiseer de optimal agent episode
    visualize_episode(maze, optimal_path, title=f"Optimal Agent Path (Reward: {optimal_reward})")

    # C. Extra Opdracht: Stochastische Omgeving
    print("\nRunning Value Iteration (Stochastic)...")
    stochastic_values, stochastic_policy = stochastic_value_iteration(maze, gamma=1.0, theta=0.01)

    # Visualiseer stochastische value function en policy
    visualize_maze(maze, stochastic_values, stochastic_policy,
                   title="Stochastic Value Function and Policy")

    # Test stochastic optimal agent
    stochastic_optimal_policy = OptimalPolicy(maze, stochastic_policy)
    stochastic_agent = Agent(maze, stochastic_optimal_policy)

    # Simuleer een episode met stochastic optimal agent (in stochastische omgeving)
    stoch_path, stoch_reward, stoch_steps = stochastic_agent.simulate_episode(stochastic=True)
    print(f"\nStochastic Optimal Agent Path Length: {stoch_steps}")
    print(f"Stochastic Optimal Agent Total Reward: {stoch_reward}")

    # Visualiseer de stochastic agent episode
    visualize_episode(maze, stoch_path,
                      title=f"Stochastic Optimal Agent Path (Reward: {stoch_reward})")

    # Vergelijk deterministische en stochastische policies
    compare_policies(maze, deterministic_values, deterministic_policy,
                     stochastic_values, stochastic_policy)

    # Analyse van de verschillen
    print("\nAnalyse van verschillen tussen deterministische en stochastische omgeving:")
    different_actions = 0
    for i in range(maze.height):
        for j in range(maze.width):
            position = (i, j)
            state = maze.get_state(position)
            if not state.is_terminal:
                if deterministic_policy[position] != stochastic_policy[position]:
                    different_actions += 1
                    det_action = deterministic_policy[position].name
                    stoch_action = stochastic_policy[position].name
                    print(f"State {position}: Deterministic: {det_action}, Stochastic: {stoch_action}")

    print(
        f"Aantal states met andere optimale acties: {different_actions} van de {maze.width * maze.height - len(maze.terminal_positions)}")


if __name__ == "__main__":
    main()