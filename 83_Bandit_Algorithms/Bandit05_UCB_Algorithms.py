import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

example = False

class GaussianSample():
    def __init__(self, mu=None, var=None, draw_option='gaussian', random_state=None):
        self.mu = mu
        self.var = var

        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.draw_option = draw_option
    
    def random_setting(self):
        self.mu = self.rng.uniform()
        self.var = self.rng.uniform()+0.5

    def show_true(self):
        print(f"(mu) {self.mu:.4f}, (var) {self.var:.4f}")
    
    def draw_gaussian(self, n_samples=1):
        if (self.mu is None or self.var is None):
            self.random_setting()
        
        if n_samples == 1:
            return self.rng.normal(self.mu, self.var)
        else:
            return self.rng.normal(self.mu, self.var, size=n_samples)

    def draw_binorm(self, n_samples=1, threshold=0.5):
        result = self.draw_gaussian(n_samples=n_samples)
        if n_samples == 1:
            return int(result >= threshold)
        else:
            return (result >= threshold).astype(int)

    def draw(self, n_samples=1):
        if self.draw_option == 'gaussian':
            return self.draw_gaussian(n_samples)
        elif self.draw_option == 'binomial':
            return self.draw_binorm(n_samples)

    def __call__(self, n_samples=1):
        return self.draw(n_samples=n_samples)


if example:
    gs1 = GaussianSample(draw_option='binomial')
    gs1.random_setting()
    # gs1.draw()


    gs2 = GaussianSample(draw_option='binomial')
    gs2.random_setting()
    # gs2.draw()


#########################################################################################
class UCB():
    def __init__(self, actions, sigma=1, delta=0.05):
        self.actions = actions
        self.sigma = sigma
        self.n_actions = len(self.actions)
        self.delta = delta
        self.t = 1

        self.cum_rewards = np.zeros(self.n_actions)
        self.cum_n = np.zeros(self.n_actions)
        
        self.batched_cum_rewards = [0] * self.n_actions
        self.batched_cum_n = [0] * self.n_actions

        self.state = 'ready'
        self.history = {}
        self.history['mu'] = []
        self.history['std'] = []
        self.history['ucb'] = []
        self.history['actions'] = []
        self.history['rewards'] = []
    
    def select_arm(self, verbose=0):
        if (self.state == 'ready' or self.state=='observe_reward'):
            if len(self.history['rewards']) == 0:
                action = np.random.choice(range(self.n_actions))

                self.history['mu'].append([0] * self.n_actions)
                self.history['std'].append([np.inf] * self.n_actions)
                self.history['ucb'].append([np.inf] * self.n_actions)
            
            else:
                prev_mu = self.history['mu'][-1]
                mu = [0] * self.n_actions
                std = [np.inf] * self.n_actions
                ucb = [np.inf] * self.n_actions

                for i in range(self.n_actions):
                    if self.cum_n[i] > 0:
                        mu[i] = (self.cum_rewards[i]-self.batched_cum_rewards[i]) / (self.cum_n[i]-self.batched_cum_n[i])
                        std[i] = np.sqrt(2 * self.sigma * np.log(1/self.delta) / self.cum_n[i])
                        ucb[i] = mu[i] + std[i]
                action = np.argmax(ucb)
                
                self.history['mu'].append(mu)
                self.history['std'].append(std)
                self.history['ucb'].append(ucb)

            self.history['actions'].append(action)
            self.state = 'select_arm'
            

            if verbose:
                print(f"action: {action}", end='\t')

    def observe_reward(self, verbose=0):
        if self.state == 'select_arm':
            
            if len(self.history['actions']) - 1 == len(self.history['rewards']):
                action = self.history['actions'][-1]
                reward = self.actions[action].draw()    ## observe reward
                
                self.history['rewards'].append(reward)
                self.state = 'observe_reward'
                self.t += 1

                self.cum_n[action] += 1
                self.cum_rewards[action] += reward
                if verbose:
                    print(f"reward: {reward}", end='\t')

    def update_params(self):
        self.batched_cum_n = [0] * self.n_actions
        self.batched_cum_rewards = [0] * self.n_actions

    def run(self, update=True, verbose=0):
        if verbose:
            print(f"(step {self.t}) ", end="")
        self.select_arm(verbose=verbose)
        self.observe_reward(verbose=verbose)
        if update:
            self.update_params()
        if verbose:
            print()

    def visualize(self, t=-1, return_plot=False):
        round = self.t+t if t < 0 else t
        mu = np.array(self.history['mu'][round-1])
        std = np.array(self.history['std'][round-1])
        Q1 = mu - std
        Q3 = mu + std

        fig, ax = plt.subplots()
        ax.set_title(f'Boxplot for UCB at {round} round')

        for i in range(self.n_actions):
            if Q3[i] < np.inf:
                box = plt.Rectangle((i+1 - 0.2, Q1[i]), 0.4, Q3[i]-Q1[i], edgecolor='black', facecolor='whitesmoke')
                ax.add_patch(box)
                ax.plot([i+1 - 0.2, i+1 + 0.2], [mu[i], mu[i]], color='black')
        ax.set_xlim(0, self.n_actions+1)
        if len((Q1-std)[Q1-std>-np.inf]) >0 and len((Q3+std)[Q3+std<np.inf]) >0:
            ax.set_ylim(np.min((Q1-std)[Q1-std>-np.inf]), np.max((Q3+std)[Q3+std<np.inf]))
        ax.set_xticks( list(range(1,self.n_actions+1)) )
        ax.set_xticklabels( list(range(self.n_actions)) )

        if return_plot:
            plt.close()
            return fig


if example:
    gs1 = GaussianSample(draw_option='binomial')
    gs2 = GaussianSample(draw_option='binomial')
    gs3 = GaussianSample(draw_option='binomial')
    actions = [gs1, gs2, gs3]

    ucb = UCB(actions)
    ucb.run(update=False)
    ucb.visualize()
    ucb.update_params()
    ucb.history.keys()

    ucb = UCB(actions)
    for _ in range(100):
        ucb.run()
    ucb.visualize()

    # ucb.history['mu']
    # ucb.t

    # ucb.cum_rewards
    # ucb.cum_n
    # ucb.history['mu']
    # ucb.history['ucb']
    # ucb.history['actions']
    # ucb.history['rewards']



from IPython.display import clear_output
import time

if example:
    gs1 = GaussianSample(draw_option='binomial')
    gs2 = GaussianSample(draw_option='binomial')
    gs3 = GaussianSample(draw_option='binomial')
    actions = [gs1, gs2, gs3]
    ucb = UCB(actions)

    vis_interval = 5
    for i in range(300):
        ucb.run(0)

        if i % vis_interval == 0: 
            ucb.visualize()
            plt.show()
            clear_output(wait=True)
            time.sleep(0.1)
