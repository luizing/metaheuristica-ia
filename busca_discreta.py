import numpy as np
import matplotlib.pyplot as plt

def perturbar(x):
    indices = (np.random.permutation(len(x)-1)+1)[:2]
    indices2 = np.roll(indices,-1)
    x[indices] = x[indices2]
    return x

def f(x, cidades):
    s = 0
    for i in range(len(x)):
        p1 = cidades[x[i]]
        p2 = cidades[x[(i+1)%len(x)]]
        # s += np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        s += np.linalg.norm(p1-p2)
    return s


def desenhar_caminhos(x, cidades, ax):
    linhas = []
    for i in range(len(x)):
        p1 = cidades[x[i]]
        p2 = cidades[x[(i+1)%len(x)]]
        if i == 0:
            linha = ax.plot([p1[0], p2[0]],[p1[1], p2[1]], c='g')
        elif i == len(x)-1:
            linha = ax.plot([p1[0], p2[0]],[p1[1], p2[1]], c='pink')
        else:
            linha = ax.plot([p1[0], p2[0]],[p1[1], p2[1]], c='k')
        linhas.append(linha[0])
    return linhas


def atualiza_caminhos(x, cidades, ax,linhas):
    for linha in linhas:
        linha.remove()
    linhas = []
    for i in range(len(x)):
        p1 = cidades[x[i]]
        p2 = cidades[x[(i+1)%len(x)]]
        if i == 0:
            linha = ax.plot([p1[0], p2[0]],[p1[1], p2[1]], c='g')
        elif i == len(x)-1:
            linha = ax.plot([p1[0], p2[0]],[p1[1], p2[1]], c='pink')
        else:
            linha = ax.plot([p1[0], p2[0]],[p1[1], p2[1]], c='k')
        linhas.append(linha[0])
    return linhas


       



N = 25

cidades = np.concatenate((
    np.array([[84, 23]]),
    np.random.uniform(10, 120,(N, 2))
))

fig = plt.figure(1)
ax = fig.add_subplot(1,2,1)
ax.scatter(cidades[:,0], cidades[:,1],c='b',edgecolors='k')

x_opt = np.concatenate((
    [0],
    np.random.permutation(N)+1
))
f_opt = f(x_opt,cidades)
ax.set_title(f"Caminho original: f(x) = {f_opt:.5f}")
linhas = desenhar_caminhos(x_opt,cidades, ax)

ax = fig.add_subplot(1,2,2)
ax.scatter(cidades[:,0], cidades[:,1],c='b',edgecolors='k')
ax.set_title(f"f(x) = {f_opt:.5f}")
linhas = desenhar_caminhos(x_opt,cidades, ax)

#Têmpera Simulada (Simulated Annealing)
T = 100
max_iteracao = 60000
historico = [f_opt]
for i in range(max_iteracao):
    x_cand = perturbar(np.copy(x_opt))
    f_cand = f(x_cand,cidades)
    P = np.exp(-((f_cand - f_opt)/T))
    if f_cand < f_opt or P >= np.random.uniform():
        f_opt = f_cand
        x_opt = x_cand
        
        # plt.pause(.5)
    historico.append(f_opt)
    T = T*0.9999
ax.set_title(f"f(x) = {f_opt:.5f}")
linhas = atualiza_caminhos(x_opt,cidades,ax,linhas)

plt.figure(2)
plt.plot(historico)
plt.xlabel("Iterações")
plt.ylabel("Valor da função")


    

plt.show()
