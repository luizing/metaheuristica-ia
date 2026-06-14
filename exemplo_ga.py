import numpy as np
import matplotlib.pyplot as plt
from algoritmos_geneticos import GeneticAlgorithm

def f(x,y):
    return np.abs(x*y*np.sin(y*np.pi/4))


inf = -1
sup = 15

ga = GeneticAlgorithm(
    f,
    (inf,sup),
    2,
    20,
    10,
    100,
    0.8,
    0.01
)

ga.search()