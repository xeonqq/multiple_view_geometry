from .algorithm import calculate_epipolar_line_on_other_image, calculate_essential_matrix

class Scene(object):
    def __init__(self, cube, cameras, renderer):
        self._cube = cube
        self._cameras = cameras
        self._renderer = renderer

    def project(self, interactive=False):
        lists_of_points_in_camera_frame = []
        lists_of_points_in_image_frame = []
        key_points_cube = self._cube.surfaces()
        for camera in self._cameras:
            points_in_camera_frame, points_in_image_frame = camera.project(key_points_cube)
            lists_of_points_in_camera_frame.append(points_in_camera_frame)
            lists_of_points_in_image_frame.append(points_in_image_frame)
            self._renderer.render_image_frame_in_camera(points_in_image_frame, camera)

        essential_matrix_cam1 = calculate_essential_matrix(self._cameras[1], self._cameras[0])
        for p in lists_of_points_in_camera_frame[1].T:
            line = calculate_epipolar_line_on_other_image(p, essential_matrix_cam1, self._cameras[0])
            self._renderer.render_epipolar_line(line, self._cameras[0], interactive)

        essential_matrix_cam0 = calculate_essential_matrix(self._cameras[0], self._cameras[1])
        for p in lists_of_points_in_camera_frame[0].T:
            line = calculate_epipolar_line_on_other_image(p, essential_matrix_cam0, self._cameras[1])
            self._renderer.render_epipolar_line(line, self._cameras[1], interactive)



