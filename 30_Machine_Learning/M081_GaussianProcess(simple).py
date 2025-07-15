import numpy as np
import matplotlib.pyplot as plt

#################################################################################################################
def rbf(x1, x2, sigma=1, l=1, const=0):
    dist = x1 - x2.T
    K = (sigma**2) * np.exp(-1/(2 * l**2) * dist**2) + const
    return K
#################################################################################################################

x = np.random.rand(10).reshape(-1,1)*6-3
y = np.random.rand(10).reshape(-1,1)

# ---------------------------------------------------------------------------------------------------------------

x_p = np.linspace(-3,3, 100).reshape(-1,1)

sigma = 1
l = 0.5

cov = rbf(x, x, sigma, l)
y_mu = cov.T @ np.linalg.inv(cov) @ y

cov_p = rbf(x, x_p, sigma, l)
cov_test = rbf(x_p, x_p, sigma, l)

y_mu_p = cov_p.T @ np.linalg.inv(cov) @ y
y_cov_p = cov_test - cov_p.T @ np.linalg.inv(cov) @ cov_p
y_std_p = np.sqrt(np.diag(y_cov_p).reshape(-1,1))


# visualize
plt.figure()
plt.scatter(x, y)
plt.plot(x_p, y_mu_p)
plt.fill_between(x_p.ravel() ,( y_mu_p - 1.96*y_std_p ).ravel(), ( y_mu_p + 1.96*y_std_p ).ravel(), alpha=0.2, color='purple')
plt.show()
# ---------------------------------------------------------------------------------------------------------------