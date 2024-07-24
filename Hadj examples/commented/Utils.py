#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 17:46:43 2023

@author: wavbrain
"""


import csv
import numpy as np
import pywt 
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile,f_classif
from sklearn import svm
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns
from scipy.integrate import simps
from scipy import signal

def normalize_segments(Segments):
    """
    This function is used to normalize a data segment (in Segments)
    """
    Segments_normalized = []
    for s in Segments:
        Min = min(min(Segments))
        Max = max(max(Segments))        
        ss = (np.array(s) - Min)  / (Max - Min)
        Segments_normalized.append(ss.tolist())
    return Segments_normalized

def GetPowersRatios(window, sf = 100):
    """
    This function calculates power spectral density (PSD) for bands of interest. 
    It also calculates some uyseful PSD ratios
    """
    win = 4 * sf
    freqs, psd = signal.welch(window, sf, nperseg=win)
    low, high = 0.5, 4.
    idx_4 = np.logical_and(freqs >= low, freqs <= high)
    low, high = 7.9, 12.
    idx_1 = np.logical_and(freqs >= low, freqs <= high)
    low, high = 30.1, 80
    idx_3 = np.logical_and(freqs >= low, freqs <= high)
    low, high = 12.1, 30
    idx_2 = np.logical_and(freqs >= low, freqs <= high)
    freq_res = freqs[1] - freqs[0]  # = 1 / 4 = 0.25
    
    power_1= simps(psd[idx_1], dx=freq_res)
    power_2 = simps(psd[idx_2], dx=freq_res)
    power_3 = simps(psd[idx_3], dx=freq_res)
    power_4 = simps(psd[idx_4], dx=freq_res)
    
    return power_1,power_2, power_4, power_3,power_1/power_4,power_3/power_4

def Calcul_gamma(X):
    """
    Jaakkola method to optimize gamma parameter for SVM classifier with RBF kernel
    """
    n,m=X.shape
    gamma=0
    distance=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            d=0
            for k in range(m):
                d=d+(X[i,k] - X[j,k])**2
            distance[i,j]=np.sqrt(d)
    gamma=1/(2*((np.median(distance))**2))
    return gamma


def GetEntropy(signal):
    """
    Calculates the entropy of a signal
    Input: the signal 
    Output: the Shannon enropy
    """
    hist, bin_edges = np.histogram(signal, density=False)
    hist = hist/len(signal)
    return entropy(hist, base=2)
    

def wrcoef(X, coef_type, coeffs, wavename, level):
    """
    Reconstructs the coefficients vector of type coef_type based on the wavelet decomposition cpeffs of a 1-D signal 
    This function is called in GetWaves
    """
    N = np.array(X).size
    a, ds = coeffs[0], list(reversed(coeffs[1:]))
    if coef_type =='a':
        return pywt.upcoef('a', a, wavename, level=level)[:N]
    elif coef_type == 'd':
        return pywt.upcoef('d', ds[level-1], wavename, level=level)[:N]
    else:
        raise ValueError("Invalid coefficient type: {}".format(coef_type))


def GetWaves(signal,Wtype='db5',level=5):
    """
    Calculates waves of interest for a signal
    Input: the signal , Wtype: wavelet type, level: decomposition level
    Output: dictionary with the different extracted waves
    """    
    coeffs  = pywt.wavedec(signal,Wtype,level)
    w_1   = wrcoef(signal, 'a', coeffs, Wtype, 5)
    w_2   = wrcoef(signal, 'd', coeffs, Wtype, 5)
    w_3    = wrcoef(signal, 'd', coeffs, Wtype, 3)
    w_4   = wrcoef(signal, 'd', coeffs, Wtype, 4)
    w_5   = wrcoef(signal, 'd', coeffs, Wtype, 2)
    Waves   = {}  
    Waves['wave_1'] = w_1
    Waves['wave_2']  = w_2
    Waves['wave_3'] = w_3
    Waves['wave_4'] = w_4
    Waves['wave_5'] = w_5
    return Waves
    
    

def GetEntropyWaves(Waves):
    """
    Calculates the entropy of each signal wave (in Waves)
    """ 
    w_1 = Waves['wave_1']
    w_2 = Waves['wave_2']
    w_3 = Waves['wave_3']
    w_4 = Waves['wave_4']
    w_5 = Waves['wave_5']
    EntropyWaves = {}
    EntropyWaves['wave_1'] = GetEntropy(w_1)
    EntropyWaves['wave_2']  = GetEntropy(w_2)
    EntropyWaves['wave_3'] = GetEntropy(w_3)
    EntropyWaves['wave_4'] = GetEntropy(w_4)
    EntropyWaves['wave_5'] = GetEntropy(w_5)
    return EntropyWaves

def GetEnergyWaves(Waves):
    """
    Calculates the energy of each signal wave (in Waves)
    """ 
    w_1 = Waves['wave_1']
    w_2  = Waves['wave_2']
    w_3 = Waves['wave_3']
    w_4 = Waves['wave_4']
    w_5 = Waves['wave_5']
    EnergyWaves = {}
    EnergyWaves['wave_1'] = np.sum(w_1**2)
    EnergyWaves['wave_2']  = np.sum(w_2**2)
    EnergyWaves['wave_3'] = np.sum(w_3**2)
    EnergyWaves['wave_4'] = np.sum(w_4**2)
    EnergyWaves['wave_5'] = np.sum(w_5**2)
    return EnergyWaves


def GetEnergyRatioWaves(Waves):
    """
    Calculates the energy ratios of each signal wave (in Waves)
    """ 
    EnergyRatioWaves = GetEnergyWaves(Waves)
    S =  sum(EnergyRatioWaves.values())
    EnergyRatioWaves['wave_1']    = EnergyRatioWaves['wave_1']  / S
    EnergyRatioWaves['wave_2']     = EnergyRatioWaves['wave_2']   / S
    EnergyRatioWaves['wave_3']    = EnergyRatioWaves['wave_3'] / S
    EnergyRatioWaves['wave_4']    = EnergyRatioWaves['wave_4'] / S
    EnergyRatioWaves['wave_5']    = EnergyRatioWaves['wave_5']  / S
    return EnergyRatioWaves


def GetWavesMin(Waves):
    """
    Calculates the minimum values of each signal wave (in Waves)
    """ 
    WavesMin = {}
    WavesMin['wave_1']    = np.min(Waves['wave_1'])
    WavesMin['wave_2']     = np.min(Waves['wave_2'])
    WavesMin['wave_3']    = np.min(Waves['wave_3'])
    WavesMin['wave_4']    = np.min(Waves['wave_4'])
    WavesMin['wave_5']    = np.min(Waves['wave_5'])
    return WavesMin

def GetWavesMax(Waves):
    """
    Calculates the maximum values of each signal wave (in Waves)
    """ 
    WavesMax = {}
    WavesMax['wave_1']    = np.max(Waves['wave_1'])
    WavesMax['wave_2']     = np.max(Waves['wave_2'])
    WavesMax['wave_3']    = np.max(Waves['wave_3'])
    WavesMax['wave_4']    = np.max(Waves['wave_4'])
    WavesMax['wave_5']    = np.max(Waves['wave_5'])
    return WavesMax

def GetWavesVar(Waves):
    """
    Calculates the variance values of each signal wave (in Waves)
    """ 
    WavesVar = {}
    WavesVar['wave_1']    = np.var(Waves['wave_1'])
    WavesVar['wave_2']     = np.var(Waves['wave_2'])
    WavesVar['wave_3']    = np.var(Waves['wave_3'])
    WavesVar['wave_4']    = np.var(Waves['wave_4'])
    WavesVar['wave_5']    = np.var(Waves['wave_5'])
    return WavesVar

def GetWavesMean(Waves):
    """
    Calculates the mean values of each signal wave (in Waves)
    """ 
    WavesMean = {}
    WavesMean['wave_1']    = np.mean(Waves['wave_1'])
    WavesMean['wave_2']     = np.mean(Waves['wave_2'])
    WavesMean['wave_3']    = np.mean(Waves['wave_3'])
    WavesMean['wave_4']    = np.mean(Waves['wave_4'])
    WavesMean['wave_5']    = np.mean(Waves['wave_5'])
    return WavesMean

    
def FeatureExtractor_Signal_OneSegment(window):
    """
    Extract features (minimum, maximum, mean, variance, entropy and energy) for one segment (window)
    """ 
    Features = []
    Features.append(np.min(window))
    Features.append(np.max(window))
    Features.append(np.mean(window))
    Features.append(np.var(window))
    Features.append(GetEntropy(window))
    Features.append(np.sum(window**2))
    return np.array(Features)
     


def FeatureExtractor_All(Segments,WType,level):
    """
    This is the main function to extract all features for one segment (window)
    """ 
    Features = []
    for segment in Segments:
        features = FeatureExtractor_Waves_OneSegment(np.array(segment),WType,level)
        features2 = FeatureExtractor_Signal_OneSegment(np.array(segment))
        Features.append(np.concatenate((features, features2), axis=0))
    return np.array(Features)

def DimensionReductor_PCA(Features=None,pca=None,Ncomponents=None):
    """
    Performs principal component analysis
    Input: Features vector (Features), PCA object if any (pca), the number of principal components (Ncomponents)
    Output: projected features vector (principalComponents), PCA object (pca)
    """ 
    if pca == None:
        pca = PCA(n_components = Ncomponents)
        principalComponents = pca.fit_transform(Features)
    else:
        principalComponents = pca.transform(Features)
    
    return principalComponents,pca
      

def ModelDesigner_OneClassSVM(KERNEL="rbf",Nu=0.1,GAMMA=0.1):
    """
    Creates a One Class SVM classifier with given parameters
    """ 
    clf = svm.OneClassSVM(kernel=KERNEL, gamma=GAMMA,nu=Nu )
    
    return clf


def ModelFitter_OneClassSVM(Features_train,classifier):
    """
    Fits a One Class SVM classifier using Features_train
    """ 
    classifier.fit(Features_train)
    
    return classifier

def ModelRuntime_OneClassSVM(classifier,Features):
    """
    Uses classifier (classifier) to make decision regarding the input features vector (Features)
    Output: classification labels (labels)
    """ 
    labels = classifier.predict(Features)
    
    return labels

def DataOrganizer_RandomCrop(Features,labels,Percentage=0.2):
    """
    Divides the features vector into two subsets: train and test according to the provided percentage
    Output: train set, test set, train labels, test labels
    """
    train, test, yTrain, yTest = train_test_split(Features,labels,test_size=Percentage)
    
    return train, test,yTrain,yTest



def FeatureExtractor_Waves_OneSegment(window,WType,level):
    """
    Extracts features related to the waves extracted from a signal (window)
    This function is called in FeatureExtractor_All
    """
    Features = []
    Waves = GetWaves(window,WType,level)
    energyRatios = GetEnergyRatioWaves(Waves)
    Mins = GetWavesMin(Waves)
    Max = GetWavesMax(Waves)
    Means = GetWavesMean(Waves)
    Vars = GetWavesVar(Waves)
    ww = list(Waves.keys())
    for w in ww:
        Features.append(GetEntropy(Waves[w]))
        Features.append(energyRatios[w])
        Features.append(Mins[w])
        Features.append(Max[w])
        Features.append(Means[w])
        Features.append(Vars[w])
    alpha_power,beta_power, delta_power, gamma_power,ratio_alpha_delta,ratio_gamma_delta = GetPowersRatios(window)
    Features.append(ratio_alpha_delta)
    Features.append(ratio_gamma_delta)
    return np.array(Features)



  