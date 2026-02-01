from manim import *
import numpy as np


class ParticleHelicalMotion3D(ThreeDScene):
    def construct(self):
        # ----------------------------
        # 相机 & 坐标系
        # ----------------------------
        self.set_camera_orientation(phi=65 * DEGREES, theta=-40 * DEGREES, zoom=1.0)
        axes = ThreeDAxes(
            x_range=[-6, 6, 1],
            y_range=[-4, 4, 1],
            z_range=[-3, 9, 1],
        )
        self.add(axes)

        # ----------------------------
        # 螺旋直线参数（可调）
        # ----------------------------
        # 轴线：经过 P0，方向为 a_hat（单位向量）
        P0 = np.array([-2.0, -1.0, 0.0])
        a = np.array([1.0, 0.4, 1.2])
        a_hat = a / np.linalg.norm(a)

        # 选一个与轴线垂直的基向量 u_hat，再构造 v_hat = a_hat × u_hat
        # 这样 (u_hat, v_hat) 就是在轴线法平面里的正交基，用来绕轴转
        helper = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(helper, a_hat)) > 0.95:
            helper = np.array([0.0, 1.0, 0.0])

        u = np.cross(a_hat, helper)
        u_hat = u / np.linalg.norm(u)
        v_hat = np.cross(a_hat, u_hat)  # 已经单位且垂直

        # 螺旋半径、螺距、角速度、沿轴速度
        R = 0.8                      # 半径
        omega = 2 * PI * 0.9         # 角速度（rad/s）= 每秒0.9圈
        v_ax = 1.2                   # 沿轴速度（units/s）

        # 动画时长
        T = 8.0

        # ----------------------------
        # 轴线可视化（可选）
        # ----------------------------
        axis_line = Line3D(
            start=P0 - 2.5 * a_hat,
            end=P0 + (v_ax * T + 2.0) * a_hat,
            thickness=0.02
        ).set_color(GRAY_B)
        self.add(axis_line)

        # ----------------------------
        # 粒子位置函数：P(t) = P0 + v_ax*t*a_hat + R*(cos(ωt)u_hat + sin(ωt)v_hat)
        # ----------------------------
        t_tracker = ValueTracker(0.0)

        def particle_pos(t: float):
            return (
                P0
                + v_ax * t * a_hat
                + R * (np.cos(omega * t) * u_hat + np.sin(omega * t) * v_hat)
            )

        # 粒子
        particle = always_redraw(lambda: Sphere(
            center=particle_pos(t_tracker.get_value()),
            radius=0.12,
            resolution=(18, 18)
        ).set_color(YELLOW))

        # 轨迹（跟随画线）
        trail = TracedPath(
            lambda: particle_pos(t_tracker.get_value()),
            stroke_width=4,
            dissipating_time=0.0,  # 0 表示不消散，完整保留轨迹
        ).set_color(BLUE)

        self.add(trail, particle)

        # ----------------------------
        # 动画
        # ----------------------------
        self.begin_ambient_camera_rotation(rate=0.10)  # 慢慢转相机，看清3D效果
        self.play(t_tracker.animate.set_value(T), run_time=T, rate_func=linear)
        self.wait(0.5)
