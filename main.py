from hanoi_tower_qlearning import HanoiTowerQLearning


def main():

    ht_arena = HanoiTowerQLearning(
        start_state=[[2, 1, 0], [], []],
        goal_state=[[], [], [2, 1, 0]],
        gamma=0.8,
        max_episodes=1000,
        epsilon_greedy=True)
    solution_steps, scores, eps_list = ht_arena.train()

    print('*** SOLUTION ***')
    for step in solution_steps:
        print(step)

    score_text = "{score:.2f};{epsilon:.2f}"
    print('*** SCORES ***')
    for score, e in zip(scores, eps_list):
        print(score_text.format(score=score, epsilon=e))


if __name__ == "__main__":
    main()
