import copy
import random as rd
import math
import numpy as np
from collections import defaultdict

class RLKey:
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def __hash__(self):
        return hash((tuple(tuple(x) for x in self.k1),tuple(tuple(x) for x in self.k2)))

    def __eq__(self, other):
        return (self.k1, self.k2) == (other.k1, other.k2)

    def __str__(self):
        return str(self.k1)+'|'+str(self.k2)

class HanoiTowerQLearning:

    def __init__(self, start_state, goal_state, gamma=0.8, max_episodes=50000, epsilon_greedy=True, min_epsilon=0.1, max_epsilon=1.0):
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
                            # disk on destination rig
                            if dest_rig[-1] > source_rig[-1]:
                                copied_rigs[i] = copied_rigs[i][:-1]
                                copied_rigs[j].append(source_rig[-1])
                                next_moves.append(copied_rigs)
                        else:
                            # no disk on destination rig
                            copied_rigs[i] = copied_rigs[i][:-1]
                            copied_rigs[j].append(source_rig[-1])
                            next_moves.append(copied_rigs)
        return next_moves

    def get_reward(self, state):
        if state == self.goal_state:
            return 100
        return 0

    def train(self):

        convergence_count = 0
        q_s_a = defaultdict(lambda: 0)
        q_s_a_prec = copy.deepcopy(q_s_a)
        episode = 1
        # rd.seed(42)
        
        scores = []
        eps_list = []
        rewards = {}
        while episode <= self.max_episodes:
            initial_state_for_this_episode = self.start_state
            score_per_episode = 0
            print('*** EPISODE '+str(episode)+' ***')
            while initial_state_for_this_episode != self.goal_state:
                next_states_for_action = self.get_next_allowed_moves(initial_state_for_this_episode)
                # k = RLKey(initial_state_for_this_episode, initial_state_for_this_episode)    
                # rewards[k] = self.get_reward(initial_state_for_this_episode)
                for next_state in next_states_for_action:
                    k = RLKey(initial_state_for_this_episode, next_state)    
                    rewards[k] = self.get_reward(next_state)

                chosen_next_state = rd.choice(next_states_for_action)
                if self.epsilon_greedy:
                    e = rd.uniform(0, 1)
                    if e > self.epsilon:
                        # action with max value from current state
                        # it's ok to randomly choose if every q is 0 because we would max on a full-0 list
                        s_a_list = {x: q_s_a[x] for x in q_s_a.keys() if x.k1 == initial_state_for_this_episode}
                        if len(s_a_list) > 0:
                            m = max(s_a_list, key=s_a_list.get)
                            chosen_next_state = m.k2
                # print(chosen_next_state)
              
                q_s1_list = {x: q_s_a[x] for x in q_s_a.keys() if x.k1 == chosen_next_state}

                if len(q_s1_list) > 0:
                    m_q_s1 = max(q_s1_list.values())
                else:
                    m_q_s1 = 0

                k_qsa = RLKey(initial_state_for_this_episode, chosen_next_state)    
                q_s_a[k_qsa] = rewards[k_qsa] + (self.gamma * m_q_s1)
                score_per_episode += q_s_a[k]
                initial_state_for_this_episode = chosen_next_state

            if q_s_a == q_s_a_prec:
                if convergence_count > int(10):
                    print('** CONVERGED **')
                    break
                else:
                    convergence_count += 1
            else:
                q_s_a_prec = copy.deepcopy(q_s_a)
                convergence_count = 0

            # epsilon update
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode)

            scores.append(score_per_episode)
            eps_list.append(self.epsilon)
            episode += 1

        solution_steps = [self.start_state]
        next_state = self.start_state
        while next_state != self.goal_state:
            candidate_next_list = {x: q_s_a[x] for x in q_s_a.keys() if x.k1 == next_state}
            m = max(candidate_next_list, key=candidate_next_list.get)
            next_state = m.k2
            solution_steps.append(next_state)
        return solution_steps, scores, eps_list

