"""

 Created on 24-Mar-21
 @author: Kiril Zelenkovski
 @topic: Homework Assigment 1

Implementation of function that does Contrast stretching on a given image. The function works with aн arbitrary
number of points and takes two arguments: image array and main list with sensitivity levels (both input and output).

The result is an image stretched on all channels.
"""
# Python imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# Plotly configuration
init_notebook_mode(connected=True)
config = {'showLink': False, 'displayModeBar': False}


# Main function for contrast stretching
def contrast_stretching(image, main_list: list):
    """
    :param image: array, image on which we apply stretching
    :param main_list:
        main_list[0] - list of input sensitivity level numbers (r1, r2, r3..)
        main_list[1] - list of output intensity level numbers (s1, s2, s3..)
    :return: array, image stretched on all channels (1 if Grayscale, 3 if RGB)
    """
    r = main_list[0]
    s = main_list[1]
    gray = False
    if len(image.shape) < 3:
        gray = True
    r.insert(0, 0)
    r.append(255)
    s.insert(0, 0)
    s.append(255)
    # Calculate slope from points
    slopes = np.diff(s) / np.diff(r)
    dd = {}
    # Combine ranges, slopes and points
    for i in range(len(r) - 1):
        # Range1: (0, 50) Range2: (51,150) ... 
        dd[(r[i], r[i + 1]) if i == 0 else (r[i] + 1, r[i + 1])] = (slopes[i], s[i])

    # Apply stretching on channels
    if gray:
        # Gray image has one channel
        image = stretch(image, dd)
    else:
        # make the stretching for all channels
        for i in range(image.shape[2]):
            image[:, :, i] = stretch(image[:, :, i], dd)

    # Return fully stretched image
    return image


def stretch(channel, dd):
    masks = []
    for k in dd:
        # Get all pixels with specified range as mask
        mask = cv2.inRange(channel, k[0], k[1])
        masks.append(mask)

    for values, mask, x in zip(dd.values(), masks, dd.keys()):
        # Get the pixels that we need to change
        px_to_modify = cv2.bitwise_and(channel, mask)
        # Get the rest of the pixels
        px_help = cv2.bitwise_or(channel, mask)
        # Create matrix rows x cols with values 255
        all_white = np.full((channel.shape[0], channel.shape[1]), 255, dtype=np.dtype('uint8'))
        non_mod_px_inv = cv2.bitwise_xor(px_help, all_white)

        # Apply stretching to the chosen pixels
        px_to_modify[px_to_modify != 0] = (px_to_modify[px_to_modify != 0] - (x[0] - 1 if x[0] > 0 else x[0])) * values[
            0] + values[1]
        # Add 255 to rest of the pixels (needed for xor)
        px_to_modify[px_to_modify == 0] = 255

        mod_values = cv2.bitwise_xor(px_to_modify, non_mod_px_inv)
        channel = mod_values

    return channel

if __name__ == '__main__':
    #         r1   r2  r3  (Input intensity level)
    r_list = [20, 120, 80]
    #         s1   s2  s3  (Output intensity level)
    s_list = [30, 200, 90]

    # Read image (np.flip is applied because it reads it upside-down in Plotly - heatmap)
    img1 = np.flip(cv2.imread("Barbara.tif", 0))
    # Apply contrast stretching
    img2 = contrast_stretching(img1, [r_list, s_list])

    # Plot using Plotly
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    # Make subplot traces of heatmaps
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(
        go.Heatmap(z=img1,
                   colorscale="Gray",
                   showscale=False,
                   autocolorscale=False,
                   name="Original image"),
        row=1, col=1)

    fig.add_trace(
        go.Heatmap(z=img2,
                   colorscale="Gray",
                   showscale=False,
                   autocolorscale=False,
                   name="Stretched image"),
        row=1, col=2)

    # Updates on traces
    fig.update_traces(overwrite=True)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    # Update layout (80% height, 80% width)
    fig.update_layout(height=(402 * 2) * 0.8,
                      width=(566 * 2) * 0.8,
                      title="\t \t \t \t \t \t \t \t \t \t \t  \t \t \t \t \t Original \t \t \t \t \t \t \t \t \t \t "
                            "\t \t \t \t \t \t \t \t \t \t \t \t \t \t \t \t \t \t \t \t \t \t \t \t \t \t \t \t \t \t"
                            " \t \t \t \t \t \t \t \t \t \t \t \t \t \t \t Stretched", )

    # Plot and save as HTML in Python script
    plot(fig, filename='1-Contrast-stretching.html', config=config)
