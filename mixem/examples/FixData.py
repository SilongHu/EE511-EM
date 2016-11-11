#!/usr/bin/env python
from bokeh.plotting import figure, output_file, show
import pandas as pd
import numpy as np
from bokeh.io import output_notebook; output_notebook()

import mixem
# prepare some data
data = pd.read_csv("faithful.csv")
#print(data.head())

# output to static HTML file
output_file("Fitting_Data_Contour.html", title="Old Faithful Data")

# create a new plot with a title and axis labels
'''
fig = figure(title="Old Faithful Data", x_axis_label="Eruption duration (minutes)", y_axis_label="Waiting time (minutes)")
fig.scatter(x=data.eruptions, y=data.waiting)
show(fig);
'''
weights, distributions, ll,iteration = mixem.em(np.array(data), [
    mixem.distribution.MultivariateNormalDistribution(np.array((2, 50)), np.identity(2)),
    mixem.distribution.MultivariateNormalDistribution(np.array((4, 80)), np.identity(2)),
])

N = 100
x = np.linspace(np.min(data.eruptions), np.max(data.eruptions), num=N)
y = np.linspace(np.min(data.waiting), np.max(data.waiting), num=N)
xx, yy = np.meshgrid(x, y, indexing="ij")
#print x,y
# Convert meshgrid into a ((N*N), 2) array of coordinates
xxyy = np.array([xx.flatten(), yy.flatten()]).T

# Compute model probabilities
p = mixem.probability(xxyy, weights, distributions).reshape((N, N))
#print p
fig2 = figure(title="Fitted Old Faithful Data", x_axis_label="Eruption duration (minutes)", y_axis_label="Waiting time (minutes)")

# Plot the grid of model probabilities -- attention: bokeh expects _transposed_ input matrix!
fig2.image(image=[p.T], x=np.min(data.eruptions), y=np.min(data.waiting), dw=np.ptp(data.eruptions), dh=np.ptp(data.waiting), palette="Spectral11")

# Plot data points
fig2.scatter(x=data.eruptions, y=data.waiting, color="#000000")

show(fig2);
