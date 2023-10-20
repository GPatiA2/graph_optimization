import g2o as g2o
import numpy as np


class DataSampler():

    def __init__(self, a, b, c, sigma):

        self.a = a
        self.b = b
        self.c = c
        self.sigma = sigma

    def sample(self, k):
        samples = []
        for i in range(k):
            x = np.random.uniform(0, 5)
            val = np.e ** (self.a * x ** 2 + self.b * x + self.c)
            samples.append((x, val + np.random.normal(0, self.sigma)))

        return samples
    
if __name__ == '__main__':

    for it in g2o.__dict__.items():
        if "Base" in it[0]:
            print(it)
    input()

    sampler = DataSampler(5,3,6,2)

    samples = sampler.sample(100)

    solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer = g2o.SparseOptimizer()
    optimizer.set_verbose(True)
    optimizer.set_algorithm(solver)
    [print(it) for it in dir(optimizer)]
    input()

    vertex = g2o.BaseEdge_3_Vector3()
    vertex.set_id(0)
    vertex.set_estimate(np.zeros(3))
    optimizer.add_vertex(vertex)

    for i in range(len(samples)):
        edge = g2o.BaseFixedSizedEdgeBaseUnaryEdge_3_Vector3_VertexSE3Expmap()
        edge.set_vertex(0, vertex)
        edge.set_measurement(samples[i])
        edge.set_information(np.identity(1))
        optimizer.add_edge(edge)

    optimizer.initialize_optimization()
    optimizer.optimize(100)

    print('Estimated parameters: ', vertex.estimate())
    print('Ground truth: ', [5,3,6])


 