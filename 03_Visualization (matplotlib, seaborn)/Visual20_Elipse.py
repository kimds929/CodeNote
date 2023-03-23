import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_ellipse(x,y, sigma=1, **kwargs):
    x_np = np.array(x).ravel()
    y_np = np.array(y).ravel()

    x_mean = x_np.mean()
    y_mean = y_np.mean()
    cov = np.cov(x_np, y_np)
    slope = cov[1,0] / cov[0,0]
    
    plt.gca().add_patch(patches.Ellipse(xy=[x_mean, y_mean], 
                                        width=np.sqrt(cov[0,0]) * sigma, 
                                        height=np.sqrt(cov[1,1]) * sigma, 
                                    angle=np.arctan(slope)*180/np.pi, 
                                    **kwargs))

draw_ellipse(gv_dropna[x1],gv_dropna[x2], sigma=3, fill=False, color=group_color[gi], alpha=0.3)