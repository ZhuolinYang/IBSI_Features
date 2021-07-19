# Intensity-Based Statistical Features
'''
Following the IBSI, there are 18 features in this category.
The features in this set do not require discretisation.
'''

import numpy as np

def get_Mean(ima):
    return np.nanmean(ima)


def get_Variance(ima):
    return np.nanvar(ima)


def get_Skewness(ima):
    var = get_Variance(ima)
    if var == 0:
        Skewness = 0
    else:
        n = ima.size
        mu = np.nanmean(ima)
        num = np.sum((ima - mu) ** 3) / n
        den = (np.sum((ima - mu) ** 2) / n) ** (3/2)
        Skewness = num / den
    return Skewness


def get_Kurtosis(ima):
    var = get_Variance(ima)
    if var == 0:
        Kurtosis = 0
    else:
        n = ima.size
        mu = np.nanmean(ima)
        num = np.sum((ima - mu) ** 4) / n
        den = (np.sum((ima - mu) ** 2) / n) ** 2
        Kurtosis = (num / den) - 3
    return Kurtosis


def get_Median(ima):
    return np.nanmedian(ima)


def get_Minimun(ima):
    return np.nanmin(ima)


def get_10Percentile(ima):
    return np.nanpercentile(ima,10)


def get_90Percentile(ima):
    return np.nanpercentile(ima,90)


def get_Maximun(ima):
    return np.nanmax(ima)


def get_InterquartileRange(ima):
    return np.nanpercentile(ima,75) - np.nanpercentile(ima,25)


def get_IntensityRange(ima):
    return np.nanmax(ima) - np.nanmin(ima)


def get_MeanAbsoluteDeviation(ima):
    n = ima.size
    mu = np.nanmean(ima)
    return np.sum(np.abs(ima - mu)) / n


def get_RobustMeanAbsoluteDeviation(ima):
    percent10 = np.nanpercentile(ima, 10)
    percent90 = np.nanpercentile(ima, 90)
    percentArray = [x for x in ima.flatten() if ((x-percent10 >= 0) and (x-percent90 <= 0))]
    percentArray = np.array(percentArray)
    n = len(percentArray)
    mu = np.nanmean(percentArray)
    return np.sum(np.abs(percentArray - mu)) / n


def get_MedianAbsoluteDeviation(ima):
    n = ima.size
    m = get_Median(ima)
    return np.sum(np.abs(ima - m)) / n


def get_CoefficientOfVariation(ima):
    sigma = get_Variance(ima) ** (1/2)
    mu = get_Mean(ima)
    return sigma / mu


def get_QuartileCoefficientOfDispersion(ima):
    percent75 = np.nanpercentile(ima,75)
    percent25 = np.nanpercentile(ima,25)
    return (percent75 - percent25) / (percent75 + percent25)


def get_Energy(ima):
    return np.sum(ima ** 2)


def get_RootMeanSquareIntensity(ima):
    n = ima.size
    return (np.sum(ima ** 2) / n) ** (1/2)


def get_FOSfeatures(ima):
    Mean = get_Mean(ima)
    Variance = get_Variance(ima)
    Skewness = get_Skewness(ima)
    Kurtosis = get_Kurtosis(ima)
    Median = get_Median(ima)
    Minimun = get_Minimun(ima)
    Percentile10 = get_10Percentile(ima)
    Percentile90 = get_90Percentile(ima)
    Maximum = get_Maximun(ima)
    Intensity_Range = get_IntensityRange(ima)
    Interquartile_Range = get_InterquartileRange(ima)
    Mean_Absolute_Deviation = get_MeanAbsoluteDeviation(ima)
    RobustMean_Absolute_Deviation = get_RobustMeanAbsoluteDeviation(ima)
    Median_Absolute_Deviation = get_MedianAbsoluteDeviation(ima)
    Coefficient_Of_Variation = get_CoefficientOfVariation(ima)
    Quartile_Coefficient_Of_Dispersion = get_QuartileCoefficientOfDispersion(ima)
    Energy = get_Energy(ima)
    Root_Mean_Square_Intensity = get_RootMeanSquareIntensity(ima)

    return Mean, Variance, Skewness, Kurtosis, Median, Minimun, Percentile10,\
           Percentile90, Maximum, Intensity_Range, Interquartile_Range,\
           Mean_Absolute_Deviation, RobustMean_Absolute_Deviation,\
           Median_Absolute_Deviation, Coefficient_Of_Variation,\
           Quartile_Coefficient_Of_Dispersion, Energy, Root_Mean_Square_Intensity

