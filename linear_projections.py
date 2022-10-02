#import utils
import torch
import torch.nn as nn
#from utils import set_default, show_scatterplot, plot_bases
from plot_lib import set_default, show_scatterplot, plot_bases
from matplotlib.pyplot import plot, title, axis, figure, gca, gcf 
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_points = 1_000
X = torch.randn(n_points,2).to(device)

x_min = -1.5
x_max = +1.5

colors = (X-x_min)/(x_max-x_min)
colors = (colors*511).short().numpy()
colors = np.clip(colors,0,511)

print(X)



