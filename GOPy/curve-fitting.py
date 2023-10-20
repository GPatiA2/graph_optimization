import g2o as g2o
import numpy as np


class CurveFittingVertex(g2o.BaseVertex_3_Vector3):

    def __init__(self):
        super(CurveFittingVertex, self).__init__()

    def setToOriginImpl(self):
        self._estimate = np.zeros(3)

    def oplusImpl(self, v):
        self._estimate += v

class CurveFittingEdge(g2o.BaseEdge_1_double):

    def __init__(self, x, measurement):
        super(CurveFittingEdge, self).__init__()
        self._measurement = measurement
        self._val = x

    def computeError(self):
        
        v = self.vertices()[0]
        x = v.estimate()
        self._error = self._measurement - np.e ** (x[0] * self._val ** 2 + x[1] * self._val + x[2])

    def linearizeOplus(self):

        v = self.vertices()[0]
        x = v.estimate()

        self.jacobianOplusXi = np.zeros((1, 3))

        y = np.e ** (x[0] * self._val ** 2 + x[1] * self._val + x[2])
        self.jacobianOplusXi[0, 0] = -self.val * self.val * y
        self.jacobianOplusXi[0, 1] = -self.val * y
        self.jacobianOplusXi[0, 2] = -y

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
            samples.append(val + np.random.normal(0, self.sigma))

        return samples
    
if __name__ == '__main__':

    sampler = DataSampler(5,3,6,2)

    samples = sampler.sample(100)

    solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer = g2o.SparseOptimizer()
    optimizer.set_verbose(True)
    optimizer.set_algorithm(solver)

    vertex = g2o.VertexSE3()
    vertex.set_id(0)
    vertex.set_estimate(g2o.Isometry3d(np.zeros(3)))
    optimizer.add_vertex(vertex)

    for i in range(len(samples)):
        edge = CurveFittingEdge(i, samples[i])
        edge.set_vertex(0, vertex)
        edge.set_measurement(samples[i])
        edge.set_information(np.identity(1))
        optimizer.add_edge(edge)

    optimizer.initialize_optimization()
    optimizer.optimize(100)

    print('Estimated parameters: ', vertex.estimate())
    print('Ground truth: ', [5,3,6])


