import numpy as np
import g2o


class BundleAjustment(object):
    def __init__(self):
        self._optimizer = g2o.SparseOptimizer()
        block_solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        optimization_algorithm = g2o.OptimizationAlgorithmLevenberg(block_solver)
        self._optimizer.set_algorithm(optimization_algorithm)

    def add_camera_parameters(self, focal_length, principle_point):
        camera_parameter = g2o.CameraParameters(focal_length, principle_point, 0)
        camera_parameter.set_id(0)
        self._optimizer.add_parameter(camera_parameter)
        return camera_parameter

    def add_pose(self, pose_id, pose, fixed=False):
        vertex_se3 = g2o.VertexSE3Expmap()
        vertex_se3.set_id(pose_id)
        vertex_se3.set_estimate(pose)
        vertex_se3.set_fixed(fixed)
        self._optimizer.add_vertex(vertex_se3)

    def add_point(self, point_id, point, fixed=False, marginalized=True):
        vertex_xyz = g2o.VertexSBAPointXYZ()
        vertex_xyz.set_id(point_id)
        vertex_xyz.set_estimate(point)
        vertex_xyz.set_marginalized(marginalized)
        vertex_xyz.set_fixed(fixed)
        self._optimizer.add_vertex(vertex_xyz)

    def add_edge(
        self,
        point_id,
        pose_id,
        measurement,
        information=np.identity(2),
        robust_kernel=g2o.RobustKernelHuber(np.sqrt(5.991)),
    ):
        edge = g2o.EdgeProjectXYZ2UV()
        edge.set_parameter_id(0, 0)
        edge.set_vertex(0, self._optimizer.vertex(point_id))
        edge.set_vertex(1, self._optimizer.vertex(pose_id))
        edge.set_measurement(measurement)  # pixel point (u,v)
        edge.set_information(information)

        edge.set_robust_kernel(robust_kernel)
        self._optimizer.add_edge(edge)

    def optimize(self, iterations=10, verbose=True):
        self._optimizer.set_verbose(verbose)
        self._optimizer.initialize_optimization()
        self._optimizer.optimize(iterations)

    def vertex_estimate(self, vertex_id):
        vertex = self._optimizer.vertex(vertex_id)
        return vertex.estimate()

    def vertex(self, vertex_id):
        return self._optimizer.vertex(vertex_id)
