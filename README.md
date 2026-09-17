# QQ飞车舞蹈箭头识别 Demo

当前 Demo 实现四项功能：

1. 对 `tests/fixtures` 中的截图进行离线箭头识别。
2. 实时截取屏幕指定区域并显示识别框。
3. 箭头序列稳定后，按照可配置的随机时间模拟方向键输入。
4. 追踪节奏条滑块，在判定区中心附近按正态分布模拟空格输入。
5. 使用现有模板检测框自动采集 YOLO 箭头图片和 8 类标签。

## 安装

建议使用项目电脑上已有的 Python 3.10：

```powershell
D:\Pyhton\python3.10.7\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 离线识别

```powershell
.\.venv\Scripts\python.exe main.py offline
```

默认识别当前玩法和 UI 对应的 `tests/fixtures` 样本。标注图片和 JSON 汇总保存在 `debug/offline`。

也可以指定图片：

```powershell
.\.venv\Scripts\python.exe main.py offline "tests\fixtures\某张截图.png"
```

## 实时观察

首次运行先框选游戏里的整条箭头区域：

```powershell
.\.venv\Scripts\python.exe main.py calibrate
```

鼠标拖动选择区域后按 Enter 保存，按 Esc 取消。随后运行：

```powershell
.\.venv\Scripts\python.exe main.py
```

不加参数时默认进入 `live` 实时模式；显式执行 `main.py live` 的效果相同。

双击 `start.vbs` 可以静默启动，只显示识别窗口，不显示 Python 控制台。`start.bat` 也已改用 `pythonw.exe`，但双击批处理时可能短暂闪过启动窗口。

## 玩法与 UI 在线切换

观察窗口顶部提供两行鼠标选项：

- 玩法：传统四键、飞车舞蹈、双人舞蹈
- UI：经典、焕新

只有识别状态为 `STOPPED` 时可以切换。运行或暂停时点击选项不会切换，并提示先停止识别。切换成功后立即保存到 `config.json`，不需要关闭程序。

第三行的 `SPLIT TRAIN/VAL` 用于选择 `Ctrl+=` 周期截图的保存集合。只能在 `STOPPED` 状态下点击，切换后会写入 `config.json`；运行或暂停期间禁止切换，避免同一轮数据混入两个集合。

`YOLO ON/OFF` 用于切换识别后端。它只在识别状态为 `STOPPED` 时允许点击：开启并加载 10 类模型后，一次推理同时识别 8 类箭头、节奏条和滑块；光标使用已标定的 ROI 相对位置。关闭后恢复传统 OpenCV 模板识别。运行或暂停过程中不能切换，避免两个识别器的连续帧结果混入同一轮。

当前版本已有“传统四键 × 经典/焕新”的完整识别素材。飞车舞蹈和双人舞蹈的选择入口与独立配置已建立，但在对应素材和规则补齐前不会允许启动识别。

也可以只为本次启动临时覆盖初始选项：

```powershell
.\.venv\Scripts\python.exe main.py --game-mode traditional_four_key --ui-mode classic
.\.venv\Scripts\python.exe main.py --game-mode traditional_four_key --ui-mode renewed
```

公共设置保存在 `config.json`；六种组合的识别配置位于 `configs/modes/<玩法>/<UI>.json`；正式模板位于 `assets/<玩法>/<UI>/templates`。

快捷键：

- `Ctrl+F7`：切换观察窗口置顶状态
- `Ctrl+F8`：重新框选箭头识别区域
- `Ctrl+F9`：开始识别
- `Ctrl+F10`：暂停／继续
- `Ctrl+F11`：停止识别并清空状态
- `Ctrl+F12`：开始／结束录制当前 ROI 画面
- `Ctrl+=`：开始／结束每 0.2 秒一次的节奏标注截图
- `Q` 或 `Esc`：关闭观察窗口并退出程序

观察窗口默认将识别画面按原内容的 55% 显示，顶部状态文字保持固定大小，不会跟随画面缩小。窗口可以通过标题栏自由拖动，也可以拖动边框调整大小。缩放比例、置顶状态和实时循环帧数上限位于 `config.json` 的 `window` 配置中；置顶状态切换后会自动保存。`max_fps` 缺省时按 `60` FPS 限制，设为 `0` 可以取消限制：

```json
"window": {
  "scale": 0.55,
  "always_on_top": true,
  "max_fps": 60
}
```

配置位于 `config.json`。如果更换了显示器或游戏分辨率，请重新执行 `calibrate`。

## ROI 视频录制

第一次按 `Ctrl+F12` 开始录制，再按一次结束并保存。无论识别处于运行、暂停还是停止状态，录制都会继续。视频默认以 30 FPS、MP4 格式保存到 `recordings` 文件夹，每次录制使用独立时间戳文件名，不会覆盖旧文件。退出程序或重新框选 ROI 时，正在录制的视频也会安全结束并保存。

```json
"recording": {
  "fps": 30,
  "codec": "mp4v",
  "output_dir": "recordings"
}
```

## YOLO 箭头数据集自动采集

实时识别处于 `RUNNING` 时，程序会把当前箭头 ROI 原图和模板识别得到的检测框保存为 YOLO 数据集。默认输出结构：

```text
datasets/yolo_arrows/
├── data.yaml
├── classes.txt
├── images/
│   ├── train/<玩法>/<UI>/
│   └── val/<玩法>/<UI>/
└── labels/
    ├── train/<玩法>/<UI>/
    └── val/<玩法>/<UI>/
```

类别固定为10类。前8类箭头由程序自动填写，最后2类需要在标注工具中补充：

```text
0 up_unpressed
1 up_pressed
2 down_unpressed
3 down_pressed
4 left_unpressed
5 left_pressed
6 right_unpressed
7 right_pressed
8 rhythm_bar
9 slider
```

光标不作为 YOLO 类别。节奏条位置稳定后，继续使用配置中的相对位置标定光标。

采集配置位于 `config.json`：

```json
"dataset": {
  "output_dir": "datasets/yolo_arrows",
  "split": "train",
  "image_extension": ".png",
  "rhythm_capture_interval_seconds": 0.2
}
```

- 按一次 `Ctrl+=` 开启节奏标注截图后，程序只在 `RUNNING` 状态下每 `0.2` 秒保存一次完整箭头 ROI；再按一次关闭。
- 周期截图文件名包含 `rhythm_capture`，同名标签会写入实时识别得到的箭头框，以及 10 类 YOLO 模型识别到的 `rhythm_bar` 和 `slider` 框。YOLO 主开关关闭时也会复用预加载模型生成这两个截图标签；正式训练前仍需人工抽查。
- 图片和标签会按玩法与 UI 风格分目录保存，例如 `images/train/traditional_four_key/classic/`；对应标签位于相同层级的 `labels` 目录。
- 采集训练集时让窗口按钮显示 `SPLIT TRAIN`。请使用另一段独立录屏或游戏场次，在停止识别后点击为 `SPLIT VAL` 再采集验证集，避免相邻帧同时进入训练集和验证集。
- `datasets` 已加入 `.gitignore`，本地采集的大图片不会误提交到仓库。
- 模板检测框适合作为初始伪标签，但正式训练前仍应抽样检查图片与同名 `.txt`，修正漏框、错框和错误的按下状态。仅使用少量模板图片本身不足以训练出能适应窗口缩放的模型，主要训练数据应来自这里保存的真实 ROI。

## 训练 YOLO 10 类模型

训练依赖单独安装，不影响现有 OpenCV 识别环境。依赖文件会从 PyTorch 官方 CUDA 13.0 软件源安装 GPU 版本；不要只执行 `pip install ultralytics`，否则可能安装 CPU 版 PyTorch：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
```

如果此前已经装成版本号带 `+cpu` 的 PyTorch，执行一次强制替换：

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall -r requirements-train.txt
```

确认训练集和验证集标签检查完成后开始训练：

```powershell
.\.venv\Scripts\python.exe main.py train
```

程序会先检查 10 类名称、图片与标签是否配对，以及 YOLO 坐标是否合法。再次训练时默认优先从现有 10 类最佳权重继续；首次扩展类别时使用原 8 类箭头权重；两者都不存在时才使用通用 `yolo26n.pt`。每次训练前都会把本地初始权重复制到 `weights/backups/`，不会删除或覆盖原权重。

如果 `images/val` 为空，程序会用固定随机种子从 `train` 生成约 `80/20` 的训练、验证文件清单，原图片和标签不会被移动。自动划分适合当前这批数据开始训练，但同一次连续录制的画面较相似，验证指标可能偏乐观；以后仍建议补充另一局独立录制的数据作为正式验证集。

默认配置为 `100` 轮、输入尺寸 `640`、批大小 `8`、早停等待 `20` 轮，并关闭会改变箭头方向语义的水平和垂直翻转增强。新增加的两个类别会重新学习，原模型的箭头和画面特征会继续迁移。

当前电脑的 RTX 3060 Laptop GPU 训练几十到几百张图片，预计纯训练约 `2～8` 分钟；首次建立缓存和 CUDA 初始化会额外耗时。最佳权重默认位于：

```text
runs/yolo_arrow/yolo26n_rhythm_10class/weights/best.pt
```

重复训练时 Ultralytics 会为新结果目录自动追加编号；程序结束时会打印本次实际目录和 `best.pt` 路径。原 8 类权重仍保留在原目录中。

实时窗口中的 YOLO 开关默认读取：

```text
runs/yolo_arrow/yolo26n_rhythm_10class/weights/best.pt
```

程序启动时会预先加载模型并执行一次 CUDA 预热，因此观察窗口第一次出现可能会比以前慢几秒；窗口出现后再开启 YOLO 或第一次正式识别时不需要重新加载。关闭 YOLO 后模型仍保留在内存中，再次开启可直接复用。开关状态会保存到 `config.json` 的 `yolo.enabled`；如果模型缺失、CUDA 不可用或推理异常，程序会关闭 YOLO 并自动回退到 OpenCV。

如需调整，可使用例如：

```powershell
.\.venv\Scripts\python.exe main.py train --epochs 120 --batch 8 --device 0
```

## 方向键输入时序

按 `Ctrl+F9` 时应先让游戏窗口处于前台。程序会绑定当时的前台窗口；输入期间一旦切换到其他窗口，本组方向键立即取消，避免输入到其他应用。

默认时序配置：

```json
"input": {
  "enabled": true,
  "auto_elevate": true,
  "reaction_delay_ms": { "min": 110, "max": 180 },
  "key_hold_ms": { "min": 25, "max": 45 },
  "inter_key_delay_ms": { "min": 35, "max": 75 },
  "clear_frames_to_rearm": 3,
  "fallback_rearm_ms": 350,
  "target_window_title_contains": ""
}
```

- `reaction_delay_ms`：一组箭头首次被发现到第一个方向键按下的随机等待。代码强制最小值不低于 110 ms。
- `key_hold_ms`：每个方向键保持按下的随机时间。
- `inter_key_delay_ms`：上一个键松开到下一个键按下的随机间隔。
- `clear_frames_to_rearm`：箭头区域连续多少帧为空后，才允许识别下一组。
- `fallback_rearm_ms`：没有捕获到紫色或空帧时，上一组完成多久后允许用不同的稳定全蓝序列判定下一组。
- `target_window_title_contains`：可选的目标窗口标题关键字；留空时绑定按下 `Ctrl+F9` 时的前台窗口。
- `enabled`：设为 `false` 可以恢复成只观察、不输入。
- `auto_elevate`：键盘输入启用且当前权限不足时，通过 UAC 自动用管理员权限重启。QQ飞车以管理员权限运行时必须保持为 `true`。

默认使用单尺度模板并要求连续两帧序列一致；校验过程发生在 110 ms 反应等待窗口内，首键不会被后续图像识别阻塞。旧组出现紫色已按下箭头或暂时消失后，新的整组蓝色箭头会被视为下一组，不要求两组之间必须出现连续空帧。

## 自动空格时序

只有稳定识别到一组箭头后，程序才会开启该轮节奏追踪；没有箭头时不会按空格。YOLO 开启时，一次推理同时输出箭头、节奏条和滑块，节奏状态机只使用第 8、9 类框；当前帧漏检时沿用已锁定的 YOLO 节奏条和短期轨迹预测，不再逐帧切换到 OpenCV。只有关闭 YOLO，或 YOLO 整体推理异常并被禁用后，才使用完整 OpenCV 流程。

经典模式使用人工统一后的节奏条标注完成光标标定。YOLO 模式下，光标中心取滤波后节奏条左边缘加上节奏条宽度的 `0.837351`，即约 `83.735%`；窗口或箭头 ROI 缩放后仍会保持相同的节奏条相对位置。OpenCV 回退仍使用原来的光标模板和搜索先验；焕新模式尚未使用这批经典 UI 数据标定。

滑块的多个位置用于作匀速直线拟合，再计算它与固定光标中心重合的时刻。方向键整组输入完成后才会启用本轮空格，且每轮只触发一次。观察窗口中的绿色细框表示已经锁定的节奏条 ROI，黄色竖线表示滑块位置，白色竖线表示光标中心。

交点预测会短期缓存。滑块接近白区时即使被人物或光圈特效遮挡，已经由匀速轨迹算出的交点时刻仍然有效；如果本轮箭头消失或等待超时，缓存会立即作废，不会带到下一轮。

三段“偏早／标准／偏晚”录像用于确认判定关系：标准样本对应两者中心基本重合；按键后的光圈仅作为反馈，通常比真实按键晚约一帧，不参与实时触发。

```json
"space": {
  "enabled": true,
  "bar_template": "assets/traditional_four_key/classic/templates/节奏条.png",
  "slider_templates": ["assets/traditional_four_key/classic/templates/节奏条滑块.png"],
  "cursor_templates": [
    "assets/traditional_four_key/classic/templates/节奏条光标.png",
    "assets/traditional_four_key/classic/templates/节奏条光标2.png",
    "assets/traditional_four_key/classic/templates/节奏条光标3.png"
  ],
  "bar_search_roi_in_arrow_roi": [0.08, 0.0, 0.84, 0.45],
  "bar_width_ratio": 0.541,
  "bar_match_threshold": 0.50,
  "bar_low_pass_alpha": 0.01,
  "slider_match_threshold": 0.62,
  "cursor_match_threshold": 0.68,
  "cursor_bar_ratio": 0.837351,
  "cursor_prior_ratio": 0.75,
  "cursor_search_radius_ratio": 0.06,
  "mean_offset_ms": 0,
  "stddev_ms": 8,
  "max_abs_offset_ms": 25,
  "key_hold_ms": { "min": 25, "max": 45 },
  "prediction_horizon_ms": 180,
  "prediction_cache_ms": 1800,
  "minimum_schedule_lead_ms": 8,
  "late_tolerance_ms": 45,
  "expire_after_directions_ms": 2200,
  "min_speed_px_per_second": 120,
  "max_speed_px_per_second": 900,
  "minimum_track_samples": 3
}
```

- `mean_offset_ms`：相对中心重合时刻的平均偏移；负数偏早，正数偏晚。
- `stddev_ms`：正态分布标准差，越小越集中在完美点附近。
- `max_abs_offset_ms`：将随机结果截断在此范围内，避免偶然出现过大的偏差。
- `bar_search_roi_in_arrow_roi`：只用于第一层节奏条定位；定位后实际识别会缩小到绿色细框内。
- `bar_width_ratio`：节奏条宽度占整个箭头 ROI 的比例，用于自动缩放模板。
- `bar_low_pass_alpha`：节奏条框的低通滤波系数；默认 `0.01`，值越小越平稳但跟随位置变化越慢，设为 `1.0` 等同于不做低通滤波。检测框跳变不会自动清空滤波状态，重新框选 ROI、停止或切换模式时才会显式重置。
- `cursor_bar_ratio`：YOLO 检测到节奏条时，光标中心相对于节奏条左边缘的位置比例；经典 UI 当前为 `0.837351`，即节奏条宽度的约 `83.735%`。
- `cursor_prior_ratio`、`cursor_search_radius_ratio`：光标在节奏条上的先验位置及左右搜索半径。
- `prediction_horizon_ms`：提前多少毫秒开始预约按键。默认值兼顾预测稳定性与系统调度余量。
- `prediction_cache_ms`：匀速交点预测可在滑块暂时被遮挡后保留多久。
- `expire_after_directions_ms`：限制本轮空格的有效期，防止漏检后误触发到后续轮次。方向键在未按下／已按下状态切换时短暂识别为空，不会再取消本轮节奏追踪；检测到下一组稳定箭头时仍会替换旧周期。

默认参数为均值 `0 ms`、标准差 `8 ms`、最大偏差 `25 ms`。如实机反馈整体偏早或偏晚，只调整 `mean_offset_ms` 即可，例如 `8` 表示整体晚 8 ms，`-8` 表示整体早 8 ms。

运行诊断记录保存在 `logs` 文件夹，每次运行生成一个形如 `2026-09-17_14-30-12-123.log` 的独立文件，自动保留最近30次日志。如果识别成功但游戏没有收到方向键，请保留游戏和程序运行状态，并检查最新日志中的 `target_bound`、`input_started`、`input_completed` 或 `input_error`。

## 当前边界

- 游戏缩放比例变化较大时，需要调整 `config.json` 中的模板匹配阈值或 `space.bar_width_ratio`。
- 节奏条被其他窗口遮挡或特效完全覆盖时，本轮会显示 `MISSED` 并跳过空格，不会盲按。
