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
from scipy.stats import beta
class BetaDistribution():
    def __init__(self, random_state=None):
        self.alpha = 1
        self.beta= 1

        self.random_state = random_state
        self.rng = np.random.RandomState()
    
    def show_params(self):
        print(f"(alpha) {self.alpha}, (beta) {self.beta}")
    
    def update(self, result):
        if result == 1:
            self.alpha += 1
        elif result == 0:
            self.beta += 1
    
    def reset(self):
        self.alpha = 1
        self.beta= 1

    def draw(self, n_samples=1):
        if n_samples == 1:
            return np.random.beta(self.alpha, self.beta)
        else:
            return np.random.beta(self.alpha, self.beta, size=n_samples)
        
    def visualize_dist(self, label=None, return_plot=False):
        x = np.linspace(0, 1, 100)
        y = beta.pdf(x, self.alpha, self.beta)

        if return_plot:
            fig = plt.figure()
        plt.plot(x, y, label=label)
        plt.fill_between(x, y, alpha=0.1)

        if return_plot:
            plt.close()
            return fig


class ThompsonSampling():
    def __init__(self, actions, dists):
        self.actions = actions
        self.n_actions = len(self.actions)
        self.dists = dists
        self.t = 1
        self.cum_n = [0] * len(self.actions)

        self.batch_actions = []
        self.batch_rewards = []
        self.state = 'ready'
        self.history = {}
        self.history['estimated_rewards'] = []
        self.history['actions'] = []
        self.history['rewards'] = []
    
    def select_arm(self, verbose=0):
        if (self.state == 'ready' or self.state == 'observe_reward') :
            sampled_list = []
            for dist in self.dists:
                sampled_list.append(dist.draw())

            action = np.argmax(sampled_list)

            self.cum_n[action] += 1
            self.history['estimated_rewards'].append(sampled_list)
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
                self.t +=1

                if verbose:
                    print(f"reward: {reward}", end='\t')
    
    def update_params(self):
        self.dists[self.history['actions'][-1]].update(self.history['rewards'][-1])
    
    def run(self, verbose=0):
        if verbose:
            print(f"(step {self.t}) ", end="")
        self.select_arm(verbose=verbose)
        self.observe_reward(verbose=verbose)
        self.update_params()
        if verbose:
            print()

    def visualize(self):
        for ei, dist in enumerate(self.dists):
            dist.visualize_dist(label=ei)
        plt.legend()


if example:
    gs1 = GaussianSample(draw_option='binomial')
    gs2 = GaussianSample(draw_option='binomial')
    actions = [gs1, gs2]
    dists = [BetaDistribution() for _ in range(len(actions))]

    ts = ThompsonSampling(actions, dists)
    ts.run(1)
    ts.visualize()

    gs1.show_true()
    gs2.show_true()



from IPython.display import clear_output
import time

if example:
    gs1 = GaussianSample(draw_option='binomial')
    gs2 = GaussianSample(draw_option='binomial')
    gs3 = GaussianSample(draw_option='binomial')
    actions = [gs1, gs2, gs3]

    dists = [BetaDistribution() for _ in range(len(actions))]

    ts = ThompsonSampling(actions,dists)

    vis_interval = 5
    for i in range(300):
        ts.run(0)

        if i % vis_interval == 0: 
            ts.visualize()
            plt.show()
            clear_output(wait=True)
            time.sleep(0.1)

