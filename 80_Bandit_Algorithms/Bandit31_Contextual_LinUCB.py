
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

example = False


# LinUCB Final Version
class LinUCB:
    def __init__(self, n_actions=2, feature_dim=0, 
            shared_theta=False, shared_context=True, 
            alpha=None, allow_duplicates=True, random_state=None):

        self.random_state = random_state
        self.rng = np.random.RandomState(self.random_state)
    
        self.n_actions = None if shared_theta and shared_context else n_actions   # Number of arms
        self.feature_dim = feature_dim  # Dimension of context vectors

        self.shared_context = shared_context
        self.shared_theta = shared_theta

        self.t = 0

        self.alpha_update_auto = True if alpha is None else False
        if self.shared_theta:
            self.A = np.identity(feature_dim)
            self.b = np.zeros(feature_dim)
            # self.theta = np.zeros(feature_dim)
            self.theta = self.rng.normal(0, 0.3, size=feature_dim)
            
            self.alpha = 1 if self.alpha_update_auto else alpha
        else:
            self.A = np.tile(np.identity(feature_dim)[np.newaxis,...],(n_actions,1,1))
            self.b = np.tile(np.zeros((1, feature_dim)),(n_actions,1))
            # self.theta = np.zeros((n_actions, feature_dim))
            self.theta = self.rng.normal(0, 0.3, size=(n_actions, feature_dim))

            self.alpha = np.ones(n_actions) if self.alpha_update_auto else alpha
        
        self.batched_contexts = np.array([])
        self.matched_action_for_batched_contexts = np.array([]).astype(int)
        self.batched_actions = np.array([]).astype(int)
        self.batched_actions_for_rewards_match = np.array([]).astype(int)
        self.batched_rewards = np.array([])

        self.history = {}
        self.history['mu'] = []
        self.history['var'] = []
        self.history['ucb'] = []
        self.history['actions'] = np.array([]).astype(int)
        self.history['rewards'] = np.array([])

        self.allow_duplicates = allow_duplicates

    def calc_mu(self, context, action=None):
        context = np.array(context).astype(float)
        if context.ndim == 1:
            context = context[np.newaxis, ...]

        mu = context @ self.theta.T

        if self.shared_theta:
            return mu.reshape(-1,1)    #(n, 1)
        else:
            if action is None:
                return mu   #(n, a)
            else:
                return mu[:,action]   # (n, 1), (n, a)
    
    def calc_var(self, context, action=None):
        context = np.array(context).astype(float)
        if context.ndim == 1:
            context = context[np.newaxis, ...]
        A_inv = np.linalg.inv(self.A)

        if self.shared_theta:
            var = np.einsum("nd,nd->n", np.einsum("nd, de->ne", context, A_inv), context)  # (n,)
            return var.reshape(-1,1)      # (n, 1)
        else:
            var = np.einsum("nad,nd->na", np.einsum("nd, ade->nae", context, A_inv), context) # (n,a)
            if action is None:
                return var  #(n, a)
            else:
                return var[:,action]   # (n, 1), (n, a)

    def predict(self, context, action=None, return_mu=True, return_var=True, return_ucb=False):
        mu = self.calc_mu(context, action)
        var = self.calc_var(context, action)
        ucb = mu + self.alpha * np.sqrt(var)

        return_elements = []
        if return_mu:
            return_elements.append(mu)
        if return_var:
            return_elements.append(var)
        if return_ucb:
            return_elements.append(ucb)
        return tuple(return_elements)

    def update_instance(self, instance_name, value, key=None, reshape_dim=None):
        if hasattr(self, instance_name):
            instance = getattr(self, instance_name)

            if key is None:
                if len(instance) == 0:
                    setattr(self, instance_name, value.reshape(-1, reshape_dim))
                else:
                    while (value.shape[-1] > instance.shape[-1]):
                        instance = getattr(self, instance_name)
                        setattr(self, instance_name, np.pad(instance, ((0,0),(0,1)), mode='constant', constant_values=np.nan))
                    setattr(self, instance_name, np.append(instance, value.reshape(-1, reshape_dim), axis=0))
            else:
                if len(instance[key]) == 0:
                    instance[key] = value.reshape(-1, reshape_dim)
                else:
                    while (value.shape[-1] > instance[key].shape[-1]):
                        instance[key] = np.pad(instance[key], ((0,0),(0,1)), mode='constant', constant_values=np.nan)
                    instance[key] = np.append(instance[key], value.reshape(-1, reshape_dim), axis=0)

    def observe_context(self, context, action=None):
        context = np.array(context).astype(float)
        if context.ndim == 1:
            context = context[np.newaxis, ...].astype(float)
        len_context = len(context)
        len_action = 1 if np.array(action).ndim == 0 else len(np.array(action))

        if (self.shared_context is False) and (action is not None) and (len_context != len_action):
            raise Exception("input action(s) must have same length with input context(s)")

        elif (self.shared_context is False) and (action is None) and (len_context != self.n_actions):
            raise Exception("context(s) must input with corresponding action(s) when sharing parameters")

        else:
            if self.shared_context:
                fill_matched_action = np.full(len_context, np.nan)
            elif (action is not None) and (len_context == len_action):
                if np.array(action).max() >= self.n_actions:
                    raise Exception(f"invalid action input (possible actions: 0~{self.n_actions-1})")
                fill_matched_action = action
            elif (action is None) and (len_context == self.n_actions):
                fill_matched_action = np.arange(self.n_actions)
            else:
                print("else condition occurs!")

            self.update_instance('batched_contexts', context, reshape_dim=self.feature_dim)
            self.matched_action_for_batched_contexts = np.append(self.matched_action_for_batched_contexts, fill_matched_action)
        return context

    def select_action(self, context=None, action=None, allow_duplicates=None, verbose=0):
        if context is not None:
            context = self.observe_context(context, action)
        allow_duplicates = self.allow_duplicates if allow_duplicates is None else allow_duplicates
        len_contexts = len(context)

        mu, var, ucb = self.predict(context, action, return_ucb=True)

        if self.shared_context:        # select action
            if self.shared_theta:
                mask = (ucb == ucb.max())
                if allow_duplicates:
                    action = np.argmax(ucb)
                else:
                    action = np.random.choice(np.flatnonzero(mask))
                self.t += 1
            else:
                mask = (ucb == np.max(ucb, axis=1, keepdims=True))
                if allow_duplicates:
                    action = np.argmax(ucb,axis=1)
                else:
                    counts = mask.sum(axis=1,keepdims=True)
                    action = np.apply_along_axis(lambda x: np.random.choice(np.flatnonzero(x)), axis=1, arr=mask)
                self.t += len_contexts
        else:       # designated action
            action = self.matched_action_for_batched_contexts
            if self.shared_theta:
                action_idx = (action,0)
            else:
                action_idx = (np.arange(len(action)), action)
            self.t += len_contexts
        
        # reset matched_action_for_batched_contexts
        self.matched_action_for_batched_contexts = np.array([]).astype(int)

        # if (self.shared_context is True) and (self.shared_theta is False):
        if self.shared_context:
            dim = len(context)  if self.shared_theta is True else self.n_actions
            self.history['mu'].append(mu)
            self.history['var'].append(var)
            self.history['ucb'].append(ucb)
        else:
            self.history['mu'].append(mu[action_idx])
            self.history['var'].append(var[action_idx])
            self.history['ucb'].append(ucb[action_idx])
        self.history['actions'] = np.append(self.history['actions'], action)

        self.batched_actions = np.append(self.batched_actions, action)
        self.batched_actions_for_rewards_match = np.append(self.batched_actions_for_rewards_match, action)

        if verbose:
            print(f"action: {action}", end='\t')
        return action

    def observe_reward(self, reward=None, reward_f=None, verbose=0):
        if len(self.batched_actions_for_rewards_match) > 0:
            reward_save = None

            if reward is not None:              # directly injected reward
                array_reward = np.array(reward)

                if array_reward.ndim == 0:      # scalar input
                    len_reward = 1
                    reward_save = array_reward.copy()
                    self.batched_rewards = np.append(self.batched_rewards, reward_save)
                    self.batched_actions_for_rewards_match = self.batched_actions_for_rewards_match[1:]

                    if (self.shared_theta is True) and (self.shared_context is True):
                        action = self.history['actions'][-1]
                        self.batched_contexts = self.batched_contexts[[action]]
                        self.history['mu'][-1] = self.history['mu'][-1][action]
                        self.history['var'][-1] = self.history['var'][-1][action]
                        self.history['ucb'][-1] = self.history['ucb'][-1][action]

                elif array_reward.ndim == 1:
                    if (len(array_reward) == self.n_actions):
                        if len(self.batched_actions_for_rewards_match) == 1:      # scalar input
                            len_reward = 1
                            reward_save = array_reward[self.batched_actions_for_rewards_match[0]]
                            self.batched_rewards = np.append(self.batched_rewards, reward_save)
                            self.batched_actions_for_rewards_match = self.batched_actions_for_rewards_match[1:]
                        else:
                            print('Confused reward argument. Transform the reward observation to (-1,1) shape.')
                    elif (self.shared_theta is True) and (self.shared_context is True):
                        len_reward = len(array_reward)

                        if (len_reward == 1) or (len_reward == len(self.batched_contexts)):
                            reward_save = array_reward.copy()

                            if len_reward == 1:
                                action = self.history['actions'][-1]
                                self.batched_contexts = self.batched_contexts[[action]]
                                self.history['mu'][-1] = self.history['mu'][-1][action]
                                self.history['var'][-1] = self.history['var'][-1][action]
                                self.history['ucb'][-1] = self.history['ucb'][-1][action]
                                self.batched_rewards = np.append(self.batched_rewards, reward_save)

                            elif len_reward == len(self.batched_contexts):
                                self.batched_actions = np.arange(len_reward)
                                self.batched_rewards = np.append(self.batched_rewards, reward_save)
                        else:
                            print('Reward must have 1 lenth or contexts length.')

                    else:
                        len_reward = len(array_reward)
                        if len_reward <= len(self.batched_actions_for_rewards_match):         # array input
                            reward_save = array_reward.copy()
                            self.batched_rewards = np.append(self.batched_rewards, reward_save)
                            self.batched_actions_for_rewards_match = self.batched_actions_for_rewards_match[len_reward:]
                        else:
                            print('Exceeds required length of reward observations.')
                            
                elif array_reward.ndim == 2:
                    len_reward = len(array_reward)
                    if (self.shared_theta is True) and (self.shared_context is True):
                        if (len_reward == 1) or (len_reward == len(self.batched_contexts)):
                            reward_save = array_reward.ravel()
                            if len_reward == 1:
                                action = self.history['actions'][-1]
                                self.batched_contexts = self.batched_contexts[[action]]
                                self.history['mu'][-1] = self.history['mu'][-1][action]
                                self.history['var'][-1] = self.history['var'][-1][action]
                                self.history['ucb'][-1] = self.history['ucb'][-1][action]
                                self.batched_rewards = np.append(self.batched_rewards, reward_save)

                            elif len_reward == len(self.batched_contexts):
                                self.batched_actions = np.arange(len_reward)
                                self.batched_rewards = np.append(self.batched_rewards, reward_save)
                        else:
                            print('Reward must have 1 lenth or contexts length.')

                    elif len_reward <= len(self.batched_actions_for_rewards_match):
                        if array_reward.shape[1] == 1:                                  # array input
                            reward_save = array_reward.ravel()
                        else:                                                           # matrix input
                            reward_save = array_reward[np.arange(len(self.batched_actions_for_rewards_match[:len_reward])), self.batched_actions_for_rewards_match[:len_reward]]

                        self.batched_rewards = np.append(self.batched_rewards, reward_save)
                        self.batched_actions_for_rewards_match = self.batched_actions_for_rewards_match[len_reward:]
                    else:
                        print('Exceeds required length of reward observations.')

            elif reward_f is not None:          # reward from functional call
                reward_list = []
                for action in self.batched_actions:
                    reward_list.append(reward_f(action))
                reward_save = np.array(reward_list)
                self.batched_rewards = np.append(self.batched_rewards, reward_save)

            self.history['rewards'] = np.append(self.history['rewards'], reward_save)
            if verbose:
                print(f"reward: {reward_save}", end='\t')
        else:
            print("Rewards corresponding to action have already been matched.")

    def update_params(self):
        if len(self.batched_contexts) == len(self.batched_actions) == len(self.batched_rewards):
            len_update_data = len(self.batched_rewards)

            if self.shared_theta:
                contexts = self.batched_contexts
                rewards = self.batched_rewards

                self.A = self.A + contexts.T @ contexts
                self.b = self.b + rewards @ contexts
                self.theta = np.linalg.inv(self.A) @ self.b

                if self.alpha_update_auto:
                    residual = np.array(self.history['rewards']) - np.array(self.history['mu'])
                    if len(residual) > 1:
                        self.alpha = residual.std()
            else:
                for action in np.unique(self.batched_actions):
                    idx_filter = (self.batched_actions == action)
                    contexts = self.batched_contexts[idx_filter]
                    rewards = self.batched_rewards[idx_filter]

                    self.A[action] = self.A[action] + contexts.T @ contexts
                    self.b[action] = self.b[action] + rewards @ contexts
                    self.theta[action] = np.linalg.inv(self.A[action]) @ self.b[action]

                    if self.alpha_update_auto:
                        action_TF = self.history['actions'] == action
                        if np.sum(action_TF) > 1:
                            if self.shared_context:
                                mu = self.history['mu'][action_TF][:, action]
                            else:
                                mu = self.history['mu'][action_TF]
                            residual = self.history['rewards'][action_TF] - mu
                            self.alpha[action] = residual.std()
            
            self.batched_contexts = np.array([])
            self.matched_action_for_batched_contexts = np.array([]).astype(int)
            self.batched_actions = np.array([]).astype(int)
            self.batched_actions_for_rewards_match = np.array([]).astype(int)
            self.batched_rewards = np.array([])

    def undo(self):
        if len(self.batched_contexts) > 0:
            len_batched = len(self.batched_contexts)
            self.t -= len_batched

            if len_batched == len(self.history['actions']):
                self.history['mu'] = self.history['mu'][:-len_batched]
                self.history['var'] = self.history['var'][:-len_batched]
                self.history['ucb'] = self.history['ucb'][:-len_batched]
                self.history['actions'] = self.history['actions'][:-len_batched]

            if len_batched == len(self.history['rewards']):
                self.history['rewards'] = self.history['rewards'][:-len_batched]

            self.batched_contexts = np.array([])
            self.matched_action_for_batched_contexts = np.array([]).astype(int)
            self.batched_actions = np.array([]).astype(int)
            self.batched_actions_for_rewards_match = np.array([]).astype(int)
            self.batched_rewards = np.array([])

    def run(self, context, reward=None, action=None, reward_f=None, update=True, allow_duplicates=None, verbose=0):
        allow_duplicates = self.allow_duplicates if allow_duplicates is None else allow_duplicates

        if verbose:
            print(f"(step {self.t}) ", end="")
        
        self.select_action(context=context, action=action, allow_duplicates=allow_duplicates, verbose=verbose)

        if (reward is not None) or (reward_f is not None):
            self.observe_reward(reward, reward_f, verbose=verbose)
        if update:
            self.update_params()
        if verbose:
            print()


if example:
    # Parameters
    alpha = 0.1  # Exploration parameter
    feature_dim = 5
    n_actions = 3

    contexts = np.random.randn(1000, feature_dim)
    true_theta = np.stack([np.random.randn(feature_dim) for _ in range(n_actions)])
    rewards = np.stack([context @ true_theta.T + + np.random.randn() for context in contexts])
    
    # one instruction pass -------------------------------------------------
    # lucb = LinUCB(n_actions, feature_dim, alpha=alpha, allow_duplicates=False)
    lucb = LinUCB(n_actions, feature_dim, shared_theta=False, shared_context=True, allow_duplicates=False)

    lucb.run(contexts[0], rewards[0], verbose=1)
    # lucb.run(contexts[:10], rewards[:10], verbose=1)
    lucb.theta

    # lucb.run(contexts[:10], rewards[:10], update=False, verbose=1)
    lucb.update_params()
    lucb.theta

    lucb.undo()
    lucb.t

    # separate instructions -------------------------------------------------
    lucb = LinUCB(n_actions, feature_dim, allow_duplicates=False)

    lucb.select_action(contexts[0])
    lucb.batched_contexts
    lucb.batched_actions
    lucb.batched_actions_for_rewards_match
    lucb.theta

    lucb.observe_reward(rewards[:2])
    lucb.batched_rewards
    lucb.batched_actions
    lucb.batched_actions_for_rewards_match
    lucb.theta

    lucb.update_params()
    lucb.theta


    # online learning with UCB -------------------------------------------------
    lucb = LinUCB(n_actions, feature_dim, allow_duplicates=False)

    for ei, (context ,reward) in enumerate(zip(contexts, rewards)):
        lucb.run(context, reward)
    lucb.theta

    action = 2
    plt.scatter( lucb.predict(contexts)[0][:,action], rewards[:,action], alpha=0.5)
    plt.show()

    for a in range(n_actions):
        plt.scatter( lucb.predict(contexts)[0][:,a], rewards[:,a], alpha=0.2, label=a)
    plt.legend()
    plt.show()
    # --------------------------------------------------------------------------------------------------------------------------------


# # Parameters
# alpha = 0.1  # Exploration parameter
# feature_dim = 5
# n_actions = 3

# contexts = np.round(np.random.randn(1000, feature_dim),3)
# true_theta = np.round(np.random.normal(size=(5,1)),3)
# rewards = np.round(contexts @ true_theta + np.random.normal(size=(len(contexts),1)),3)

# contexts1 = np.random.randn(1000, feature_dim)
# contexts2 = np.random.randn(1000, feature_dim)

# true_theta0 = np.random.normal(size=(5,1))
# rewards1 = contexts1 @ true_theta0 + np.random.normal(size=(len(contexts1),1))
# rewards2 = contexts1 @ true_theta0 + np.random.normal(size=(len(contexts2),1))

# # ----------------------------------------------------------------
# lucb1 = LinUCB(n_actions=3, feature_dim=feature_dim, shared_theta=True, shared_context=True, allow_duplicates=False)
# lucb2 = LinUCB(n_actions=3, feature_dim=feature_dim, shared_theta=False, shared_context=True, allow_duplicates=False)
# lucb3 = LinUCB(n_actions=3, feature_dim=feature_dim, shared_theta=True, shared_context=False, allow_duplicates=False)
# lucb4 = LinUCB(n_actions=3, feature_dim=feature_dim, shared_theta=False, shared_context=False, allow_duplicates=False)


# lucb1.run(contexts[:10], rewards[:10])
# lucb2.run(contexts[:10], rewards[:10])
# lucb3.run(contexts[:10], rewards[:10], action=np.random.randint(0,3,size=10))
# lucb4.run(contexts[:10], rewards[:10], action=np.random.randint(0,3,size=10))
# lucb1.theta
# lucb2.theta
# lucb3.theta
# lucb4.theta

# lucb1.alpha
# lucb2.alpha
# lucb3.alpha
# lucb4.alpha


# #----------------------------------------------------------------
# # lucb1
# # lucb2
# # lucb3
# # lucb4


# lucb1 = LinUCB(n_actions=3, feature_dim=feature_dim, shared_theta=True, shared_context=True, allow_duplicates=False)
# lucb2 = LinUCB(n_actions=3, feature_dim=feature_dim, shared_theta=False, shared_context=True, allow_duplicates=False)
# lucb3 = LinUCB(n_actions=3, feature_dim=feature_dim, shared_theta=True, shared_context=False, allow_duplicates=False)
# lucb4 = LinUCB(n_actions=3, feature_dim=feature_dim, shared_theta=False, shared_context=False, allow_duplicates=False)

# idx = np.random.randint(0,1000, size=10)
# contexts_input = contexts[idx]
# actions_input = np.random.randint(0,3, size=10)


# lucb1.observe_context(contexts_input, action=actions_input)
# lucb2.observe_context(contexts_input, action=actions_input)
# lucb3.observe_context(contexts_input, action=actions_input)
# lucb4.observe_context(contexts_input, action=actions_input)

# # lucb1.observe_context(contexts_input[2], action=actions_input[2])
# # lucb2.observe_context(contexts_input[2], action=actions_input[2])
# # lucb3.observe_context(contexts_input[2], action=actions_input[2])
# # lucb4.observe_context(contexts_input[2], action=actions_input[2])

# lucb1.select_action(verbose=1)
# lucb2.select_action(verbose=1)
# lucb3.select_action(verbose=1)
# lucb4.select_action(verbose=1)

# lucb1.observe_reward(rewards[6])
# lucb2.observe_reward(rewards[idx])
# lucb3.observe_reward(rewards[idx])
# lucb4.observe_reward(rewards[idx])

# # lucb1.undo()
# # lucb2.undo()
# # lucb3.undo()
# # lucb4.undo()
# mu2, var2, ucb2
# # mu1, var1, ucb1 = lucb1.predict(lucb1.batched_contexts, return_ucb=True)
# # mu2, var2, ucb2 = lucb2.predict(lucb2.batched_contexts, return_ucb=True)
# # mu3, var3, ucb3 = lucb3.predict(lucb3.batched_contexts, return_ucb=True)
# # mu4, var4, ucb4 = lucb4.predict(lucb4.batched_contexts, return_ucb=True)

# lucb1.batched_contexts
# lucb2.batched_contexts
# lucb3.batched_contexts
# lucb4.batched_contexts

# lucb1.matched_action_for_batched_contexts
# lucb2.matched_action_for_batched_contexts
# lucb3.matched_action_for_batched_contexts
# lucb4.matched_action_for_batched_contexts

# lucb1.history['actions']
# lucb2.history['actions']
# lucb3.history['actions']
# lucb4.history['actions']

# lucb1.batched_actions
# lucb2.batched_actions
# lucb3.batched_actions
# lucb4.batched_actions

# lucb1.batched_actions_for_rewards_match
# lucb2.batched_actions_for_rewards_match
# lucb3.batched_actions_for_rewards_match
# lucb4.batched_actions_for_rewards_match

# lucb1.batched_rewards
# lucb2.batched_rewards
# lucb3.batched_rewards
# lucb4.batched_rewards

# lucb1.history['rewards']
# lucb2.history['rewards']
# lucb3.history['rewards']
# lucb4.history['rewards']




# ucb4[np.arange(10),lucb4.matched_action_for_batched_contexts]

# lucb1.theta.shape   # (d,)
# lucb1.A.shape   # (d, d)
# lucb1.b.shape   # (d, 1)

# lucb2.theta.shape   # (a, d)
# lucb2.A.shape   # (a, d, d)
# lucb2.b.shape   # (d, a)

# lucb3.theta.shape   # (d,)
# lucb3.A.shape   # (d, d)
# lucb3.b.shape   # (d, 1)

# lucb4.theta.shape   # (a, d)
# lucb4.A.shape   # (a, d, d)
# lucb4.b.shape   # (d, a)


# lucb1.observe_context(contexts[:3])
# lucb2.observe_context(contexts[:3])
# lucb3.observe_context(contexts[:3])
# lucb4.observe_context(contexts[:3])


# lucb1.observe_context(contexts[0], action=5)
# lucb2.observe_context(contexts[0], action=5)
# lucb3.observe_context(contexts[0], action=1)
# lucb4.observe_context(contexts[0], action=2)