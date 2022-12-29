import copy
import random as rd
import math
import numpy as np
from collections import defaultdict


class HanoiTowerQLearning:

    def __init__(self, start_state, goal_state, gamma=0.8, max_episodes=50000, min_epsilon=0.1, max_epsilon=1.0, epsilon_greedy=True):
        self.start_state = start_state
        self.goal_state = goal_state
        self.gamma = gamma
        self.max_episodes = max_episodes

        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.decay_rate = 0.02
        self.epsilon = self.max_epsilon
        self.epsilon_greedy = epsilon_greedy

    def get_next_allowed_moves(self, starting_state):
        next_moves = []
        for i, source_rig in enumerate(starting_state):

            if len(source_rig) > 0:
                for j, dest_rig in enumerate(starting_state):
                    copied_rigs = copy.deepcopy(starting_state)
                    if i != j:
                        # no point in moving on same rig
                        if len(dest_rig) > 0:
                            if dest_rig[-1] > source_rig[-1]:
                                copied_rigs[i] = copied_rigs[i][:-1]
                                copied_rigs[j].append(source_rig[-1])
                                next_moves.append(copied_rigs)
                        else:
                            copied_rigs[i] = copied_rigs[i][:-1]
                            copied_rigs[j].append(source_rig[-1])
                            next_moves.append(copied_rigs)
        return next_moves

    def get_reward(self, state):
        if state == self.goal_state:
            return 100
        return 0

    def train(self, verbose=False):

        possible_initial_states = set()
        possible_initial_states.add(str(self.start_state))

        GAMMA = self.gamma
        GOAL_STATE = self.goal_state
        MAX_EPISODES = self.max_episodes

        convergence_count = 0
        q_s_a = defaultdict(lambda: 0)
        q_s_a_prec = q_s_a
        episode = 1
        # rd.seed(42)

        scores = []
        eps_list = []
        rewards = {}
        while episode <= MAX_EPISODES:
            initial_state_for_this_episode = str(self.start_state)
            score_per_episode = 0
            print('*** EPISODE '+str(episode)+' ***')
            while eval(initial_state_for_this_episode) != GOAL_STATE:
                possible_next_states_for_this_state = self.get_next_allowed_moves(
                    eval(initial_state_for_this_episode))
                rewards[initial_state_for_this_episode+"|"+str(initial_state_for_this_episode)
                        ] = self.get_reward(initial_state_for_this_episode)
                for next_state in possible_next_states_for_this_state:
                    rewards[initial_state_for_this_episode+"|"+str(next_state)
                            ] = self.get_reward(next_state)

                # chosen_next_state = str(rd.choice(
                #    possible_next_states_for_this_state))
                if self.epsilon_greedy:
                    e = rd.uniform(0, 1)

                    if e < self.epsilon:
                        chosen_next_state = str(rd.choice(
                            possible_next_states_for_this_state))
                    else:
                        # action with max value from current state
                        s_a_list = {x: q_s_a[x] for x in q_s_a.keys() if x.startswith(
                            initial_state_for_this_episode+"|")}
                        m = max(
                            s_a_list, key=s_a_list.get)
                        chosen_next_state = m.split('|')[1]
                else:
                    chosen_next_state = str(rd.choice(
                        possible_next_states_for_this_state))
                # print(chosen_next_state)

                q_s1_list = {x: q_s_a[x] for x in q_s_a.keys() if x.startswith(
                    chosen_next_state+"|")}

                if len(q_s1_list) > 0:
                    m_q_s1 = max(q_s1_list.values())
                else:
                    m_q_s1 = 0

                q_s_a[initial_state_for_this_episode + "|" +
                      chosen_next_state] = rewards[initial_state_for_this_episode + "|" + chosen_next_state] + (GAMMA * m_q_s1)
                score_per_episode += q_s_a[initial_state_for_this_episode + "|" +
                                           chosen_next_state]
                initial_state_for_this_episode = chosen_next_state

            if q_s_a == q_s_a_prec:
                if convergence_count > int(100):
                    print('** CONVERGED **')
                    done = True
                    break
                else:
                    convergence_count += 1
            else:
                q_s_a_prec = q_s_a
                convergence_count = 0

            # epsilon update
            self.epsilon = self.min_epsilon + \
                (self.max_epsilon - self.min_epsilon) * \
                np.exp(-self.decay_rate * episode)

            scores.append(score_per_episode)
            eps_list.append(self.epsilon)
            episode += 1

        # print(q_s_a)

        c = 0
        print('*** SCORES ***')
        for score, e in zip(scores, eps_list):
            print(str(score).replace(".", ",") +
                  ";"+str(e*100).replace(".", ","))
        print(self.start_state)
        next_state = str(self.start_state)
        while next_state != str(self.goal_state):
            candidate_next_list = {x: q_s_a[x] for x in q_s_a.keys() if x.startswith(
                next_state+"|")}
            m = max(
                candidate_next_list, key=candidate_next_list.get)
            next_state = m.split("|")[1]
            print(next_state)
            c += 1
