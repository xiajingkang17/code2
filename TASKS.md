# 轨迹稳定与自适应绘图改进任务清单

说明：每个任务完成后先由你验证，通过后再进行下一个。

## 任务清单（逐项完成）

- [ ] **T1：`follow_track` 支持 `start:"auto"` + 投影起点（消除起步瞬移）**
  - 内容：在 `template/flow.py` 中加入“轨道投影”函数；当 `s_start/s_end` 或 `start/end` 传入 `"auto"` 时，自动以当前 `anchor` 或物体位置投影到轨道作为起点/终点。
  - 验证：
    1) 用一个滑块初始不在轨道起点的示例（可新建 `examples/demo_follow_track_auto.json`）。  
    2) 运行渲染后，第一帧不再“瞬移回起点”，轨迹连续无抖动。

- [ ] **T2：`follow_track` 加入运动进度 `profile/s(t)`（非匀速）**
  - 内容：`visual_transform.params` 支持 `profile`（如 `constant_accel`/`constant_decel`/`keyframes`），在 updater 内将 `alpha -> t -> s(t)`，实现匀加速/匀减速或关键帧插值。
  - 验证：
    1) 新建或修改示例，让同一路径“匀速 vs 匀加速”对比。  
    2) 观察轨迹点间距随时间变化（匀加速应逐渐增大）。

- [ ] **T3：`track` 圆弧段解析化（消除曲面角度跳变）**
  - 内容：扩展 `visuals/library/tracks.py` 的 `segments`，支持 `arc` 解析段（连续切向/法向、弧长），不再完全依赖折线采样。
  - 验证：
    1) 用圆弧轨道让滑块沿轨迹旋转。  
    2) 观察滑块角度变化是否连续（无明显“折线式跳角”）。

- [ ] **T4：引入 `diagram_scale`（每题自适应尺度，量化到 1/2/5×10^n）**
  - 内容：在 `visuals/compiler.py` 或渲染入口加入 `diagram_scale` 处理：物理坐标统一除以 `u` 后绘制；`u` 自动量化；支持断轴/省略（可先做最小版本：量化 + 缩放）。
  - 验证：
    1) 两个数值尺度差异很大的题目，画面都能“占满画布”且比例不漂。  
    2) 标注能提示 `1 unit = u m` 或关键长度保持可读。

- [ ] **T5：LLM1 输出结构化“绘图规划”并与 LLM2 对接**
  - 内容：在 `plan/llm_solver.py` 中加入 `ModelSpec/DiagramSpec` 的结构化输出规范（静态受力图 + 轨迹约束 + 分段 `s(t)` + `diagram_scale`）。
  - 验证：
    1) LLM1 产出的 `solution.txt` 含清晰的绘图结构段落。  
    2) LLM2 能稳定生成对应的 `plan.json` 视觉与动画字段。

- [ ] **T6：调试叠层（轨道、anchor、法向、轨迹 trace）可开关**
  - 内容：加入 `DEBUG_VISUAL` 或类似开关，渲染时可显示轨道、anchor 点、法向箭头、轨迹 trace。
  - 验证：
    1) 开关打开时可见调试元素；关闭时画面干净。  
    2) 调试元素与物体运动一致（不漂移）。

---
如果你确认这个清单，我就从 **T1** 开始动手，并在每个任务完成后等待你验证
