import matplotlib.pyplot as plt


class CameraImageRenderer(object):
    def __init__(self, cameras_with_color, show_image_frame, show_epipolar_lines):
        self._show_image_frame = show_image_frame
        self._show_epipolar_lines = show_epipolar_lines
        self._fig, self._axes = plt.subplots(1, len(cameras_with_color))
        self._camera_to_ax = {camera: ax for ax, camera in zip(self._axes, cameras_with_color.keys())}
        self._cameras_with_color = cameras_with_color
        self._fig.suptitle("keypoints in image frame")

    def render_image_frame_in_camera(self, points_in_image_frame, camera):
        if self._show_image_frame:
            self._camera_to_ax[camera].set_xlim(0, camera.image_resolution[0])
            self._camera_to_ax[camera].set_ylim(0, camera.image_resolution[1])
            self._camera_to_ax[camera].set_title("image frame in camera {}".format(camera.name))
            self._camera_to_ax[camera].plot(
                points_in_image_frame[0, :], points_in_image_frame[1, :], "o", color=self._cameras_with_color[camera]
            )

    def render_epipolar_line(self, line_end_points, camera, interactive=False):
        if self._show_epipolar_lines:
            self._camera_to_ax[camera].plot(line_end_points[:, 0], line_end_points[:, 1])
            if interactive:
                plt.waitforbuttonpress(0.2)
                plt.draw()
