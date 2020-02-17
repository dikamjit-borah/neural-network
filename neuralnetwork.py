import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fileData = pd.read_csv('filename')
fileDataConverted = pd.DataFrame(fileData)
weight1 = []
weight2 = []
weight3 = []

hidden1 = []
hidden2 = []
hidden3 = []

bias2 = []
bias3 = []
bias4 = []


input_nodes = features?
hidden1_nodes = 
hidden2_nodes =
output_nodes = 

w1 = np.random.randn(hidden1_nodes, input_nodes)


output = []

def sigmoid(x):
	return (1/(1 + np.exp(-x)))
	
def rectified(x):
	return max(0.0, x)
	
def feedForward:
    y1 = np.dot(w1, inputColoumn) + bias2
    hidden1 = sigmoid(y1)
    
    y2 = np.dot(w2, hidden1) + bias3
    hidden2 = sigmoid(y2)
    
    output = np.dot(w3, hidden2) +bias4
    return hidden3
    
