from collections import defaultdict
import os.path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from skimage import color, feature


def per_pixel(img, y, x):
    """Process each pixel of the grey-scale image, returning relevant
    key-value pairs"""
    yield ("present", 1 if img[y][x] < 1 else 0)
    yield ("horiz", 1 if (x > 0 and x < img.shape[1] - 1 and img[y][x-1] < 1
                          and img[y][x] < 1 and img[y][x+1] < 1) else 0)
    yield ("vert", 1 if (y > 0 and y < img.shape[0] - 1 and img[y-1][x] < 1
                         and img[y][x] < 1 and img[y+1][x] < 1) else 0)
    x_range, y_range = [0], [0]
    if x != 0:
        x_range.append(-1)
    if x < img.shape[1] - 1:
        x_range.append(1)
    if y != 0:
        y_range.append(-1)
    if y < img.shape[0] - 1:
        y_range.append(1)

    grey_neighbors = 0
    for i in x_range:
        for j in y_range:
            grey_neighbors += img[y+j][x+i]
    avg_grey = float(grey_neighbors) / len(x_range) / len(y_range)
    yield ("diff", abs(img[y][x] - avg_grey))


def featurize(img_name):
    """Load an image and convert it into a dictionary of features"""
    img = plt.imread(os.path.join('stimuli', img_name + '.png'))
    height, width, _ = img.shape
    features = defaultdict(int)
    for y in range(height):
        for x in range(width):
            features['red'] += img[y][x][0]
            features['green'] += img[y][x][1]
            features['blue'] += img[y][x][2]
            features['alpha'] += img[y][x][3]

    grey = color.rgb2grey(img)
    for y in range(height):
        for x in range(width):
            for key, value in per_pixel(grey, y, x):
                features[key] += value

    # Normalize over image size
    for key, value in features.items():
        features[key] = float(value) / height / width

    features['blob'] = feature.blob_dog(grey).shape[0]
    features['corners'] = feature.corner_peaks(
        feature.corner_harris(grey)).shape[0]
    return features


def relevant_features(spike_counts, img_names):
    """Return statistically significant correlations between features and
    spike counts"""
    img_features = {name: featurize(name) for name in set(img_names)}
    per_row = pd.DataFrame()
    for key in img_features[img_features.keys()[0]]:
        per_row[key] = img_names.map(lambda img_name:
                                     img_features[img_name][key])

    per_row['rg'] = per_row['red'] * per_row['green']
    per_row['rb'] = per_row['red'] * per_row['blue']
    per_row['gb'] = per_row['green'] * per_row['blue']
    per_row['rgb'] = per_row['red'] * per_row['green'] * per_row['blue']
    per_row['r/g'] = per_row['red'] / per_row['green']
    per_row['r/b'] = per_row['red'] / per_row['blue']
    per_row['g/b'] = per_row['green'] / per_row['blue']
    per_row['r+g'] = per_row['red'] + per_row['green']
    per_row['r+b'] = per_row['red'] + per_row['blue']
    per_row['g+b'] = per_row['green'] + per_row['blue']
    per_row['g-b'] = per_row['green'] - per_row['blue']
    per_row['b+r-g'] = per_row['blue'] + per_row['red'] - per_row['green']
    per_row['white'] = per_row['red'] + per_row['green'] + per_row['blue']
    per_row['zeros'] = np.zeros(per_row.shape[0])
    per_row['ones'] = per_row['zeros'] + 1
    per_row['random'] = np.random.rand(per_row.shape[0])
    per_row['perfect'] = spike_counts

    correlations = []
    for key in per_row.keys():
        corr, p_val = scipy.stats.spearmanr(per_row[key], spike_counts)
        if p_val <= .05 / len(per_row.keys()):
            correlations.append((abs(corr), corr, p_val, key))
    return sorted(correlations)


def calculate_spike_counts(data):
    """Given raw data, create a data frame with spike counts"""
    data = pd.DataFrame({
        'target': data['stim_names'], 'spikes': data['spk_times'],
        'stimon': data['stimon'], 'stimoff': data['stimoff']})
    spike_counts = pd.Series()
    for _, row in data.iterrows():
        spike_count = sum((row['spikes'] >= row['stimon'])
                          & (row['spikes'] <= row['stimoff']))
        spike_counts = np.append(spike_counts, spike_count)
    data['spike_count'] = spike_counts
    return data


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python runner.py data.npy")
    else:
        data = np.load(sys.argv[1])[()]
        data = calculate_spike_counts(data)
        labels, corr = [], []
        for el in relevant_features(data['spike_count'], data['target']):
            print(el)
            if el[3] != 'perfect':
                labels.append(el[3])
                corr.append(el[0] * 100)

        plt.bar(range(len(corr)), corr, align='center')
        plt.title("Statistically Significant Correlations")
        plt.xticks(range(len(corr)), labels)
        plt.ylabel("Correlation Coefficient (%)")
        plt.xlabel("Feature")
        plt.show()
