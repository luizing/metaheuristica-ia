import numpy as np
import matplotlib.pyplot as plt


class GeneticAlgorithm:
    def __init__(self, f, constraints, p, N, nbits, max_gen, pc, pm):
        self.f = f
        self.inf = constraints[0]
        self.sup = constraints[-1]
        self.p = p
        self.N = N
        self.max_gen = max_gen
        self.nb = nbits
        self.pc = pc
        self.pm = pm
        self.P = np.random.randint(0,2,size=(N, nbits*p))
        self.evaluate_fitness()


        fig = plt.figure(1)
        self.ax = fig.add_subplot(projection='3d')
        x = np.linspace(self.inf,self.sup)
        X,Y = np.meshgrid(x,x)
        self.ax.plot_surface(X,Y, self.f(X,Y),
                             cmap='gray',edgecolors='k',lw=.1, alpha=.3)
        self.points = self.ax.scatter(self.P_real[:,0], self.P_real[:,1], self.fitness,
                        c='teal', marker="D",s=90)
        plt.pause(.5)
    def evaluate_fitness(self):
        P_b = [np.split(self.P[i,:],self.p)for i in range(self.N)]
        self.P_real = np.array([[self.phi(P_b[i][0]),self.phi(P_b[i][1])] for i in range(self.N)])
        self.fitness = np.array([self.f(*self.P_real[i]) for i in range(self.N)])
        bp=1

    def phi(self,b):
        s = 0
        for i in range(self.nb):
            s += b[self.nb-i-1]*2**i
        return self.inf + (self.sup-self.inf)/(2**self.nb-1)*s
    
    def selection(self,nst = 2):
        self.S = np.empty((0, self.nb * self.p))
        for i in range(self.N):
            indexes = np.random.permutation(self.N)[:nst]
            winner_index = np.argmax(self.fitness[indexes])
            self.S = np.vstack((
                self.S,
                self.P[indexes[winner_index],:]
            ))
        self.P = np.copy(self.S)

    def crossover(self):
        for i in range(0,self.N,2):
            if np.random.uniform() <= self.pc:
                c1 = np.split(np.copy(self.P[i,:]),self.p)
                c2 = np.split(np.copy(self.P[i+1, :]),self.p)
                index = np.random.randint(1,self.nb-1)                
                for j in range(len(c1)):
                    c1[j][index:], c2[j][index:] = np.copy(c2[j][index:]) , np.copy(c1[j][index:])
                self.P[i,:] = np.concatenate((c1[0],c1[1]))
                self.P[i+1,:] = np.concatenate((c2[0],c2[1]))

    def mutation(self):
        for i in range(self.N):
            for j in range(self.nb * self.p):
                if np.random.uniform() <= self.pm:
                    self.P[i,j] = 1 if self.P[i,j]==0 else 0
    def plot_points(self):
        self.points.remove()
        self.points = self.ax.scatter(self.P_real[:,0], self.P_real[:,1], self.fitness,
                        c='teal', marker="D",s=90)
        # plt.pause(.5)
    def search(self):
        gen = 0
        best_fitness = [np.max(self.fitness)]
        worst_fitness = [np.min(self.fitness)]
        mean_fitness = [np.mean(self.fitness)]
        while gen < self.max_gen:
            #Seleção
            self.selection()
            #Recombinação
            self.crossover()
            #Mutação
            self.mutation()
            self.evaluate_fitness()
            best_fitness.append(np.max(self.fitness))
            worst_fitness.append(np.min(self.fitness))
            mean_fitness.append(np.mean(self.fitness))
            self.plot_points()
            gen+=1
        plt.figure(2)
        plt.plot(best_fitness,c='g')
        plt.plot(worst_fitness,c='r')
        plt.plot(mean_fitness,c='yellow')
        plt.show()

       