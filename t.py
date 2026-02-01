from manim import *
import numpy as np
import math

def ZH(s, size=34, font="Microsoft YaHei"):
    # Windows 一般都有微软雅黑；如果你机器没有，把 font 参数删掉也能跑（用默认字体）
    return Text(s, font=font, font_size=size)

def ZH_MATH(zh: str, math_expr: str, size_zh=34, size_math=36, font="Microsoft YaHei"):
    return VGroup(
        Text(zh, font=font, font_size=size_zh),
        MathTex(math_expr, font_size=size_math),
    ).arrange(RIGHT, buff=0.15)



class Example2_MotionLogic(Scene):
    def construct(self):
        # ----------------------------
        # 题目参数（按题干）
        # ----------------------------
        g = 10
        theta = 30 * DEGREES
        M = 1.0
        m = 1.5
        mu = np.sqrt(3) / 3
        x0 = 6.4      # A右端到挡板初始距离（m）
        v0 = 6.0      # 初速度沿斜面向下（m/s）

        # 关键推导中会用到的量（题目数值非常“整”）
        sin_t = np.sin(theta)        # 1/2
        cos_t = np.cos(theta)        # sqrt(3)/2
        a_free = -g * sin_t          # 取“沿斜面向上”为正，则重力分量加速度为 -5

        # 摩擦力大小：f = mu*m*g*cos(theta)
        f = mu * m * g * cos_t       # = 7.5
        # 发生相对滑动且B相对A向下滑时：对A是“向下”的摩擦 -> A加速度更向下
        # 在本题关键段里，A的加速度为：aA = (-Mg sinθ - f) / M = -5 - 7.5 = -12.5
        aA_slide = a_free - f / M    # = -12.5

        # (1) A撞挡板前速度：v1^2 = v0^2 + 2 g sinθ x0 => v1=10
        v1 = math.sqrt(v0**2 + 2 * g * sin_t * x0)  # 10
        # 从初态到第一次碰撞的时间：v = v0 + a t（注意方向：向下是负，初速度 -v0）
        # 取向上为正：vA(0)=-v0, a=a_free
        t_hit1 = ( -v1 - (-v0) ) / a_free   # = (-10+6)/(-5)=0.8
        # 碰后A速度反向（弹性反弹）：+v1
        # 在“相对滑动段”里，每个往返周期：A从u=0出发，v=+10，a=-12.5，回到u=0
        # u(t)=10t-6.25t^2=0 => t=0 或 1.6
        period = 2 * v1 / (v1 - (-v1))  # 只是占位，不用
        t_cycle = 1.6

        # (2) 第一次碰后A离挡板最远距离：发生在vA=0时：t= v1/12.5 = 0.8
        t_to_max = v1 / (-aA_slide)      # 0.8
        z_m = v1**2 / (2 * (-aA_slide))  # 4

        # (3) 第三次碰撞时恰好分离 -> B在第三次碰撞瞬间滑到A右端
        # 每次“碰后到下一次碰撞”的1.6s里，B的加速度为0、速度恒为 -10（向上为正）
        # B相对A每周期“吃掉”16m，总共两周期到第三次碰撞吃掉32m -> L=32m
        L = 32.0
        total_slip = 32.0
        deltaE = f * total_slip  # 240J

        # ----------------------------
        # 画面几何（把“米”缩放到屏幕坐标）
        # ----------------------------
        unit = 0.18  # 1m -> 0.18个屏幕单位（你可以调大/调小）
        wall_pt = np.array([4.0, -2.0, 0.0])

        down_dir = rotate_vector(RIGHT, -theta)   # 斜面向下方向（屏幕上一般是右下）
        up_dir = -down_dir                        # 斜面向上方向（屏幕上一般是左上）

        # 取一个“法线方向”用于把物体抬离斜面（保证在斜面“上方”）
        n1 = rotate_vector(up_dir, PI/2)
        n2 = rotate_vector(up_dir, -PI/2)
        normal_dir = n1 if n1[1] > n2[1] else n2  # y更大的那个当“上方”

        def P(u_meters: float):
            """u为沿斜面向上的距离（m），返回屏幕坐标点"""
            return wall_pt + (u_meters * unit) * up_dir

        # 斜面、挡板
        incline = Line(P(-2), P(60), stroke_width=8).set_color(GRAY_B)
        wall = Line(
            wall_pt - 1.2 * unit * normal_dir,
            wall_pt + 1.8 * unit * normal_dir,
            stroke_width=10
        ).set_color(WHITE)

        self.play(Create(incline), Create(wall), run_time=1.2)

        # ----------------------------
        # “时间”追踪器（用它驱动整段运动）
        # ----------------------------
        t = ValueTracker(0.0)

        # A右端位置 uA(t)：沿斜面向上为正，挡板处u=0
        def uA(tval: float):
            if tval <= t_hit1:
                # 初段：A与B一起做匀加速运动（a=-5），从u=x0到u=0
                return x0 + (-v0) * tval + 0.5 * a_free * tval**2
            elif tval < (t_hit1 + 2 * t_cycle):
                # 从第一次碰撞后到第三次碰撞前：每个周期从u=0出发再回到u=0
                # 这是“B相对A向下滑”的滑动段里：A用a=-12.5
                tt = tval - t_hit1
                tau = tt % t_cycle
                return 0.0 + (v1) * tau + 0.5 * aA_slide * tau**2
            else:
                # 第三次碰撞后：B已分离，A只受重力分量（a=-5）从u=0出发
                tau = tval - (t_hit1 + 2 * t_cycle)
                return 0.0 + (v1) * tau + 0.5 * a_free * tau**2

        def vA(tval: float):
            if tval <= t_hit1:
                return (-v0) + a_free * tval
            elif tval < (t_hit1 + 2 * t_cycle):
                tt = tval - t_hit1
                tau = tt % t_cycle
                return (v1) + aA_slide * tau
            else:
                tau = tval - (t_hit1 + 2 * t_cycle)
                return (v1) + a_free * tau

        # B的位置：碰撞后关键段里B速度恒为 -10（向上为正），直到第三次碰撞恰好到A右端
        def uB(tval: float):
            if tval <= t_hit1:
                # 初段：B在A左端（相对不滑），uB = uA + L
                return uA(tval) + L
            elif tval < (t_hit1 + 2 * t_cycle):
                # 从第一次碰撞后到第三次碰撞：B做匀速（a=0），v=-10
                # 第一次碰撞瞬间：A右端u=0，B在左端u=L
                return L + (-v1) * (tval - t_hit1)
            else:
                # 分离后：为了“看得清楚”，让B从A右端飞离（示意）
                return 0.0  # 其沿斜面坐标固定在挡板附近，后面用额外偏移画“飞离”

        # ----------------------------
        # 画木板A、物块B（用updater持续跟随位置）
        # ----------------------------
        board_h = 0.35
        block_size = 0.35

        angle_board = math.atan2(up_dir[1], up_dir[0])

        board = RoundedRectangle(
            width=L * unit,
            height=board_h,
            corner_radius=0.08
        ).set_fill(BLUE_E, opacity=1).set_stroke(WHITE, width=2)
        board.rotate(angle_board)

        block = Square(side_length=block_size).set_fill(YELLOW, opacity=1).set_stroke(BLACK, width=2)
        block.rotate(angle_board)

        label_A = Tex("A", font_size=28).set_color(WHITE)
        label_B = Tex("B", font_size=28).set_color(BLACK)

        def board_center(tval: float):
            # A右端在uA，板中心在 uA + L/2
            return P(uA(tval) + L/2)

        def right_end_point(tval: float):
            return P(uA(tval))

        def left_end_point(tval: float):
            return P(uA(tval) + L)

        def block_center(tval: float):
            if tval < (t_hit1 + 2 * t_cycle):
                # B还在板上：放在uB，并抬到板“上方”
                base = P(uB(tval))
                return base + normal_dir * (board_h/2 + block_size/2)
            else:
                # 分离后：从挡板处“飞离”以示意分离（不是严格求碰撞细节）
                tau = tval - (t_hit1 + 2 * t_cycle)
                base = P(0.0) + normal_dir * (board_h/2 + block_size/2)
                return base + normal_dir * (0.9 * tau) + down_dir * (0.25 * tau)

        def labelA_pos(tval: float):
            return board_center(tval) + normal_dir * 0.25

        def labelB_pos(tval: float):
            return block_center(tval) + normal_dir * 0.25

        board.add_updater(lambda m: m.move_to(board_center(t.get_value())))
        block.add_updater(lambda m: m.move_to(block_center(t.get_value())))
        label_A.add_updater(lambda m: m.move_to(labelA_pos(t.get_value())))
        label_B.add_updater(lambda m: m.move_to(labelB_pos(t.get_value())))

        self.add(board, block, label_A, label_B)

        # 轨迹（可选：看得更清楚）
        trailA = TracedPath(lambda: right_end_point(t.get_value()), stroke_width=3).set_color(BLUE_C)
        trailB = TracedPath(lambda: block_center(t.get_value()), stroke_width=3).set_color(YELLOW_C)
        self.add(trailA, trailB)

        # ----------------------------
        # 题目信息 & 分问文字
        # ----------------------------
        info = VGroup(
            Tex(r"$\theta=30^\circ,\ g=10$", font_size=30),
            Tex(r"$M=1\,kg,\ m=1.5\,kg,\ \mu=\frac{\sqrt{3}}{3}$", font_size=30),
            Tex(r"$x_0=6.4m,\ v_0=6m/s$", font_size=30),
        ).arrange(DOWN, aligned_edge=LEFT).to_corner(UL)
        self.play(FadeIn(info), run_time=0.8)

        # x0 标注（初态）
        brace_x0 = BraceBetweenPoints(wall_pt, right_end_point(0.0), buff=0.15).set_color(WHITE)
        x0_text = Tex(r"$x_0=6.4m$", font_size=28).next_to(brace_x0, normal_dir)
        self.play(GrowFromCenter(brace_x0), FadeIn(x0_text), run_time=0.8)

        # ----------------------------
        # (1) 运动：从初态到第一次碰撞
        # ----------------------------
        q1 = ZH_MATH("（1）求 A 撞挡板前速度", r"v_1").to_corner(UR)
        self.play(FadeIn(q1), run_time=0.5)

        self.play(t.animate.set_value(t_hit1), run_time=2.8, rate_func=linear)
        self.play(Indicate(wall), run_time=0.35)

        eq1 = MathTex(
            r"v_1^2=v_0^2+2g\sin\theta \,x_0",
            r"\Rightarrow v_1=10\,m/s"
        ).arrange(DOWN, aligned_edge=LEFT).scale(0.9).to_edge(RIGHT).shift(DOWN*0.8)
        self.play(FadeIn(eq1), run_time=0.6)
        self.play(FadeOut(brace_x0), FadeOut(x0_text), run_time=0.5)
        self.wait(0.4)

        # ----------------------------
        # (2) 第一次碰后：A 反弹远离挡板到最大距离 z_m
        # ----------------------------
        self.play(FadeOut(q1), run_time=0.3)
        q2 = ZH_MATH("（2）第一次碰后，A 右端到挡板最大距离", r"z_m").to_corner(UR)
        self.play(FadeIn(q2), run_time=0.5)

        # 到达最远点（碰后0.8s）
        t_max_global = t_hit1 + t_to_max
        self.play(t.animate.set_value(t_max_global), run_time=2.0, rate_func=linear)

        # 标注 z_m
        brace_z = BraceBetweenPoints(wall_pt, right_end_point(t_max_global), buff=0.15).set_color(WHITE)
        z_text = Tex(r"$z_m=4m$", font_size=28).next_to(brace_z, normal_dir)
        self.play(GrowFromCenter(brace_z), FadeIn(z_text), run_time=0.8)

        eq2 = MathTex(
            r"v_A: 10\to 0,\ a_A=5+7.5=12.5",
            r"\Rightarrow z_m=\frac{10^2}{2\cdot 12.5}=4m"
        ).arrange(DOWN, aligned_edge=LEFT).scale(0.8).next_to(eq1, DOWN, aligned_edge=LEFT)
        self.play(FadeIn(eq2), run_time=0.6)
        self.wait(0.5)

        # ----------------------------
        # (3) 继续：第二次、第三次碰撞；第三次碰撞瞬间B到A右端 -> 分离
        # ----------------------------
        self.play(FadeOut(q2), run_time=0.3)
        q3 = VGroup(
            ZH("（3）第三次碰撞时恰好分离：求", 34),
            MathTex(r"L", font_size=36),
            ZH("与", 34),
            MathTex(r"\Delta E", font_size=36),
        ).arrange(RIGHT, buff=0.15).to_corner(UR)
        self.play(FadeIn(q3), run_time=0.5)

        # 走到第二次碰撞
        t_hit2 = t_hit1 + t_cycle
        self.play(t.animate.set_value(t_hit2), run_time=2.2, rate_func=linear)
        self.play(Indicate(wall), run_time=0.35)

        # 走到第三次碰撞（分离时刻）
        t_hit3 = t_hit1 + 2 * t_cycle
        self.play(t.animate.set_value(t_hit3), run_time=2.2, rate_func=linear)
        self.play(Indicate(wall), run_time=0.35)

        # 分离后再稍微演示一下
        self.play(t.animate.set_value(t_hit3 + 0.8), run_time=1.6, rate_func=linear)

        # 公式与结论：L=32, ΔE=240J
        # 公式与结论：L=32, ΔE=240J（中文用 Text，公式用 MathTex）
        eq3 = VGroup(
            VGroup(
                Text("每次（碰后到下次碰撞）时长：", font="Microsoft YaHei", font_size=26),
                MathTex(r"1.6\,\mathrm{s}", font_size=30),
            ).arrange(RIGHT, buff=0.12),

            VGroup(
                Text("相对滑移：", font="Microsoft YaHei", font_size=26),
                MathTex(r"\Delta s=10\times 1.6=16\,\mathrm{m}", font_size=30),
            ).arrange(RIGHT, buff=0.12),

            VGroup(
                Text("到第三次碰撞共两段：", font="Microsoft YaHei", font_size=26),
                MathTex(r"L=16+16=32\,\mathrm{m}", font_size=30),
            ).arrange(RIGHT, buff=0.12),

            VGroup(
                Text("系统损失机械能：", font="Microsoft YaHei", font_size=26),
                MathTex(r"\Delta E=(\mu m g\cos\theta)\cdot 32=7.5\times 32=240\,\mathrm{J}", font_size=30),
            ).arrange(RIGHT, buff=0.12),
        ).arrange(DOWN, aligned_edge=LEFT).scale(0.9).to_edge(LEFT).shift(DOWN*1.8)

        self.play(FadeIn(eq3), run_time=0.8)

        self.wait(1.0)

        # 收尾淡出
        self.play(FadeOut(VGroup(eq1, eq2, eq3, q3, brace_z, z_text)), run_time=0.8)
        self.wait(0.2)
