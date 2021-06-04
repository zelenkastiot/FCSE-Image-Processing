"""

 Created on 03-Apr-21
 @author: Kiril Zelenkovski
 @topic: Homework Assigment 2

 Implement Compass operator for edge detection:
    - Calculate and display result of each filter (Figure 1)
    - Calculate and display result from combination of all filters (Figure 2)
    - Tested on different values for threshold (Figure 3-Image-Segmentation)

 You can load results directly by opening the html renders in a browser. The location of the figures is:

    Figure 1:    ../.figures/fig1_filter_images_direction.html
    Figure 2:    ../.figures/fig2_original_adaptive.html
    Figure 3-Image-Segmentation:    ../.figures/fig3_threshold_slider.html

"""

# Python3 imports
import math
import plotly.graph_objects as go
import plotly.tools as tls
from plotly.offline import plot, iplot, init_notebook_mode
from IPython.core.display import display, HTML
import numpy as np
import cv2
import matplotlib.pyplot as plt

init_notebook_mode(connected=True)
config = {'showLink': False, 'displayModeBar': False}


# Create function for applying filter on image
def make_filtered_image(image, filterTemp):
    filtered_img = cv2.filter2D(image, -1, filterTemp)
    filtered_img = abs(filtered_img)
    return filtered_img / np.amax(filtered_img[:])


# Read image
img = cv2.imread("Barbara.tif", 0)

# Calculate filter directions:
n = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype='float32')  # North
ne = np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]], dtype='float32')  # North-East
e = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype='float32')  # East
se = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]], dtype='float32')  # South-East
s = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype='float32')  # South
sw = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]], dtype='float32')  # South-West
w = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype='float32')  # West
nw = np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]], dtype='float32')  # North-West

# Combine filters in list
filters = [n, ne, e, se, s, sw, w, nw]

# Create labels for plot traces
labels = ["North", "North-East", "East", "South-East", "South", "South-West", "West", "North-West"]

# Finally, apply all filters on the image
filtered = []
for filterVar in filters:
    filtered.append(make_filtered_image(img, filterVar))

# -_-_-_-_-_-_-_-_-_-_-_-_ Figure 1: Filtered images based on different filter direction -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

data = []
for i, img in enumerate(filtered):
    current_trace = go.Heatmap(z=np.flip(img),
                               colorscale="Gray",
                               showscale=False,
                               autocolorscale=False,
                               name=labels[i] + " trace",
                               visible=False)
    data.append(current_trace)

data[0]['visible'] = True

# Setup the layout of the figure
layout = go.Layout(
    updatemenus=[
        dict(
            active=0,
            x=-0.1,
            y=1.1,
            yanchor="top",
            xanchor="left",
            buttons=list([
                dict(label=labels[0],
                     method="update",
                     args=[{"visible": [True, False, False, False, False, False, False, False]},
                           {'annotations': [
                               dict(text="Filter direction: ",
                                    showarrow=False,
                                    x=0.25,
                                    y=1.15,
                                    yref="paper",
                                    align="left")]}]),
                dict(label=labels[1],
                     method="update",
                     args=[{"visible": [False, True, False, False, False, False, False, False]},
                           {'annotations': [
                               dict(text="Filter direction: ",
                                    showarrow=False,
                                    x=0.25,
                                    y=1.15,
                                    yref="paper",
                                    align="left")]}]),
                dict(label=labels[2],
                     method="update",
                     args=[{"visible": [False, False, True, False, False, False, False, False]},
                           {'annotations': [
                               dict(text="Filter direction: ",
                                    showarrow=False,
                                    x=0.25,
                                    y=1.15,
                                    yref="paper",
                                    align="left")]}]),
                dict(label=labels[3],
                     method="update",
                     args=[{"visible": [False, False, False, True, False, False, False, False]},
                           {'annotations': [
                               dict(text="Filter direction: ",
                                    showarrow=False,
                                    x=0.25,
                                    y=1.15,
                                    yref="paper",
                                    align="left")]}]),
                dict(label=labels[4],
                     method="update",
                     args=[{"visible": [False, False, False, False, True, False, False, False]},
                           {'annotations': [
                               dict(text="Filter direction: ",
                                    showarrow=False,
                                    x=0.25,
                                    y=1.15,
                                    yref="paper",
                                    align="left")]}]),
                dict(label=labels[5],
                     method="update",
                     args=[{"visible": [False, False, False, False, False, True, False, False]},
                           {'annotations': [
                               dict(text="Filter direction: ",
                                    showarrow=False,
                                    x=0.25,
                                    y=1.15,
                                    yref="paper",
                                    align="left")]}]),
                dict(label=labels[6],
                     method="update",
                     args=[{"visible": [False, False, False, False, False, False, True, False]},
                           {'annotations': [
                               dict(text="Filter direction: ",
                                    showarrow=False,
                                    x=0.25,
                                    y=1.15,
                                    yref="paper",
                                    align="left")]}]),
                dict(label=labels[7],
                     method="update",
                     args=[{"visible": [False, False, False, False, False, False, False, True]},
                           {'annotations': [
                               dict(text="Filter direction: ",
                                    showarrow=False,
                                    x=0.25,
                                    y=1.15,
                                    yref="paper",
                                    align="left")]}])

            ]),
            direction="down"
        )],
    title="1<sup>st</sup> Comparison",
    title_x=0.9,
    annotations=[
        dict(text="Filter direction: ",
             showarrow=False,
             x=0.25,
             y=1.15,
             yref="paper",
             align="left")],
    height=620,
    width=460,
    plot_bgcolor='#fff',
    xaxis=dict(range=[0, np.shape(img)[1]],
               mirror=True,
               ticks='outside',
               showline=True,
               linecolor='#000',
               tickfont=dict(size=11)),
    yaxis=dict(range=[0, np.shape(img)[0]],
               mirror=True,
               ticks='outside',
               showline=True,
               linecolor='#000',
               tickfont=dict(size=11)),
    font=dict(size=11.5),
    margin=go.layout.Margin(l=50,
                            r=50,
                            b=60,
                            t=110))

# Save figure traces with layout
fig = dict(data=data, layout=layout)

# Plot figure inline in Jupyter notebook
# iplot(fig, config=config)

# Plot and save as HTML in Python script
plot(fig, filename='.figures/fig1_filter_images_direction.html', config=config)
display(HTML('.figures/fig1_filter_images_direction.html'))

# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

# -_-_-_-_-_-_-_-_-_-_-_-_- Figure 2: Original vs Edge total (no thresh) vs Adaptive Threshold -_-_-_-_-_-_-_-_-_-_-_-_-
data = []

img = cv2.imread("Barbara.tif", 0)

# calculate total edge sum
edge_total = np.maximum.reduce(filtered)

trace1 = go.Heatmap(z=np.flip(img),
                    colorscale="Gray",
                    showscale=False,
                    autocolorscale=False,
                    name="Original image (Gray)",
                    visible=True)

trace2 = go.Heatmap(z=np.flip(edge_total),
                    colorscale="Gray",
                    showscale=False,
                    autocolorscale=False,
                    name="Combined total (no threshold)",
                    visible=False)

gaussian = cv2.adaptiveThreshold(src=img,
                                 maxValue=255,
                                 adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                 thresholdType=cv2.THRESH_BINARY,
                                 blockSize=17,
                                 C=6)

trace3 = go.Heatmap(z=np.flip(gaussian),
                    colorscale="Gray",
                    showscale=False,
                    autocolorscale=False,
                    name="Mean threshold",
                    visible=False)

data2 = [trace1, trace2, trace3]

# Setup the layout of the figure
layout2 = go.Layout(
    updatemenus=[
        dict(
            active=0,
            x=-0.1,
            y=1.1,
            yanchor="top",
            xanchor="left",
            buttons=list([
                dict(label="Original image",
                     method="update",
                     args=[{"visible": [True, False, False]},
                           {'annotations': [
                               dict(text="Show trace: ",
                                    showarrow=False,
                                    x=0.25,
                                    y=1.15,
                                    yref="paper",
                                    align="left")]}]),
                dict(label="Combined total (No threshold)",
                     method="update",
                     args=[{"visible": [False, True, False]},
                           {'annotations': [
                               dict(text="Show trace: ",
                                    showarrow=False,
                                    x=0.25,
                                    y=1.15,
                                    yref="paper",
                                    align="left")]}]),
                dict(label="Adaptive threshold",
                     method="update",
                     args=[{"visible": [False, False, True]},
                           {'annotations': [
                               dict(text="Show trace: ",
                                    showarrow=False,
                                    x=0.25,
                                    y=1.15,
                                    yref="paper",
                                    align="left")]}])
            ]),
            direction="down"
        )],
    title="2<sup>nd</sup> Comparison",
    title_x=0.9,
    annotations=[
        dict(text="Show trace: ",
             showarrow=False,
             x=0.25,
             y=1.15,
             yref="paper",
             align="left")],
    height=620,
    width=460,
    plot_bgcolor='#fff',
    xaxis=dict(range=[0, np.shape(img)[1]],
               mirror=True,
               ticks='outside',
               showline=True,
               linecolor='#000',
               tickfont=dict(size=11)),
    yaxis=dict(range=[0, np.shape(img)[0]],
               mirror=True,
               ticks='outside',
               showline=True,
               linecolor='#000',
               tickfont=dict(size=11)),
    font=dict(size=11.5),
    margin=go.layout.Margin(l=50,
                            r=50,
                            b=60,
                            t=110))

# Save figure traces with layout
fig2 = dict(data=data2, layout=layout2)

# Plot figure inline in Jupyter notebook
# iplot(fig2, config=config)

# Plot and save as HTML in Python script
plot(fig2, filename='.figures/fig2_original_adaptive.html', config=config)
display(HTML('.figures/fig2_original_adaptive.html'))

# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

# -_-_-_-_-_-_-_-_-_-_-_ Figure 3-Image-Segmentation: Slider for better comparison of different threshold values -_-_-_-_-_-_-_-_-_-_-_-_-_

# Read image
img = cv2.imread("Barbara.tif", 0)

# calculate total edge sum
edge_total = np.maximum.reduce(filtered)

# Create trace list for fig3, add slider labels list
data3 = []
slider_labels = []

# Calculate all threshold values / heatmaps accordingly
for i in np.arange(0.05, 1, 0.05):
    _, thresh = cv2.threshold(edge_total, i, 1, cv2.THRESH_BINARY)
    trace = go.Heatmap(z=np.flip(thresh),
                       colorscale="Gray",
                       showscale=False,
                       autocolorscale=False,
                       name=f"T: {round(i, 2)}",
                       visible=False)
    slider_labels.append(round(i, 2))
    data3.append(trace)

# Set first trace visible
data3[0]['visible'] = True

# Create steps and slider
steps = []
for i in range(0, len(data3)):
    step = dict(
        method='restyle',
        args=['visible', [False] * 19],
        label=slider_labels[i])

    step['args'][1][i] = True
    steps.append(step)

sliders = [dict(
    active=0,
    currentvalue={'prefix': "Threshold: <b>"},
    pad={"t": 80, "b": 10},
    steps=steps
)]

# Setup the layout of the figure
layout3 = go.Layout(
    title="3-Image-Segmentation<sup>rd</sup> Comparison",
    title_x=0.9,
    sliders=sliders,
    height=710,
    width=460,
    plot_bgcolor='#fff',
    xaxis=dict(range=[0, np.shape(img)[1]],
               mirror=True,
               ticks='outside',
               showline=True,
               linecolor='#000',
               tickfont=dict(size=11)),
    yaxis=dict(range=[0, np.shape(img)[0]],
               mirror=True,
               ticks='outside',
               showline=True,
               linecolor='#000',
               tickfont=dict(size=11)),
    font=dict(size=11.5),
    margin=go.layout.Margin(l=50,
                            r=50,
                            b=60,
                            t=110))

# Save figure traces with layout
fig3 = dict(data=data3, layout=layout3)

# Plot figure inline in Jupyter notebook
# iplot(fig3, config=config)

# Plot and save as HTML in Python script
plot(fig3, filename='.figures/fig3_threshold_slider.html', config=config)
display(HTML('.figures/fig3_threshold_slider.html'))
# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
