from hanoi_tower_qlearning import HanoiTowerQLearning


def main():

    ht_arena = HanoiTowerQLearning(
        start_state=[[3, 2, 1, 0], [], []],
        goal_state=[[], [], [3, 2, 1, 0]],
        gamma=0.8,
        max_episodes=1000,
        epsilon_greedy=True)
    ht_arena.train(verbose=True)
    # print(state_space)


if __name__ == "__main__":
    main()
