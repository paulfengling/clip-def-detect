# wafer 少样本训练与缺陷画框

这个项目可以做 **正常 / 异常二分类、异常区域定位、由异常区域生成检测框**。当前训练标签只有 `0` 和 `1`，不会直接识别“划痕 / 颗粒 / 污染”等缺陷类型。`class_name` 表示产品对象，wafer 数据统一使用英文 `wafer`；缺陷类型可以放在子目录中用于采样与分别统计效果。

AA-CLIP 的原始目标是零样本异常检测：先学习区分正常与异常的文本表示，再让图像块特征与它们对齐。这里用目标 wafer 的少量正常图、异常图和 mask 训练适配器，属于**有监督少样本迁移**。只有正常图片，或者只有图像级缺陷类型标签，不能完整运行当前有监督分割训练。方法出处见 [AA-CLIP 论文](https://arxiv.org/abs/2503.06661) 与 [官方代码](https://github.com/Mwxinnn/AA-CLIP)。

## 1. 晶圆表面 / 显微镜照片

按你确认的场景，以下流程使用**晶圆表面或显微镜实拍图 + 真实缺陷区域的二值 mask**。标注划痕、颗粒、污染等缺陷本身；正常图片保留真实纹理、边缘、反光等容易误检的情况，训练与部署使用相同成像和预处理方式。wafer map 的失效模式分类属于另一类任务。

当前模型会把输入转成 RGB 并缩放为正方形；灰度图片可以输入，但 16-bit 图像应先用固定、明确的强度范围转换为 8-bit，训练和推理保持一致。极细缺陷在整图缩小时可能消失；可先提高分辨率，或统一裁成局部 patch 并标注 patch mask。当前入口没有自动滑窗拼接，patch 检测框属于 patch 自身的坐标。

## 2. 环境与模型权重

本机的 `.venv` 环境已安装好依赖，直接使用 `.venv/bin/python` 即可。已使用真实 bottle 数据跑通 8 张训练图、224 分辨率、Apple MPS 上文本 1 轮 + 视觉 1 轮，输出位于 `ckpt/bottle_runtime_smoke/`，训练记录在其 `train.log`。这验证了本机训练流程；目前尚无真实 wafer 数据，尚未训练或测量 wafer 模型精度。README 中原有 Windows / bottle 结果属于既有记录，也不是本次 wafer 评估。

在其他机器首次安装时，可在项目根目录这样创建环境；Windows 将 `.venv/bin/python` 换成 `.venv\Scripts\python.exe`：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

后面的命令直接使用本机 `.venv/bin/python`，无需激活环境；Windows 请替换为 `.venv\Scripts\python.exe`。需要 NVIDIA GPU 训练时，先安装适合显卡的 PyTorch / torchvision CUDA 版本，再安装其余依赖。训练、校准、测试和预测均可通过 `--device auto`、`cpu`、`mps`、`cuda` 选择设备；`auto` 按 CUDA → Apple MPS → CPU 的顺序选择可用设备。CPU 可以验证流程，ViT-L/14 的实际训练与推理较慢。

项目需要 `model/ViT-L-14-336px.pt`。该文件是 CLIP 主模型；`ckpt/*/text_adapter.pth` 与 `image_adapter.pth` 是另行训练的适配器，两者都需要。主模型来源和下载映射见 [OpenAI CLIP 官方实现](https://github.com/openai/CLIP/blob/main/clip/clip.py)。已有 bottle 适配器适合做接口演示，不能代表 wafer 模型效果。

## 3. 数据、标注与最小数量

整理成下面的目录；`defect` 下缺陷类型子目录可选：

```text
data/wafer_dataset/
  wafer/
    good/
      normal_001.png
      normal_002.png
    defect/
      scratch/
        defect_001.png
        defect_001.json       # 可选：Labelme 标注
      particle/
        defect_002.png
    masks/
      scratch/
        defect_001_mask.png
      particle/
        defect_002_mask.png
```

- 图片支持 PNG、JPG / JPEG、BMP、TIF / TIFF；推荐统一可解码、同一尺度和成像条件的格式。
- 正常图：`label=0`，`mask_path` 为空或省略。不要把有未标注缺陷的图片放进正常类。
- 异常图：`label=1`，必须有 mask。mask 是单通道 PNG，背景为 `0`，缺陷为 `255`（也支持 `0/1`），不能全黑。
- mask 宽高与**原图完全一致**，坐标对齐。只标缺陷本身，多个缺陷取并集；不要标整片 wafer，也不要使用彩色热力图作为 mask。
- `defect/scratch/a.jpg` 对应 `masks/scratch/a_mask.png`。同一子目录不要同时存在 `a.jpg` 与 `a.png`，它们会争用同一个 mask 名。
- 图像级多类别标签、YOLO / COCO 框标注不能直接用于当前训练。矩形可以转成弱 mask，但会包含正常背景，分割质量受影响；优先使用真实缺陷轮廓。

### 按缺陷类型标注，保留原始类别信息

在 Labelme 中沿每个真实缺陷区域画多边形，并给**每个区域**填写对应类型，例如：

| 区域类型 | Labelme 标签 |
| --- | --- |
| 划痕 | `scratch` |
| 颗粒 | `particle` |
| 污染 | `contamination` |

一张图有多个缺陷就画多个区域。同一片同时有划痕和颗粒时，分别标为 `scratch` 和 `particle`，保存在同一张图片对应的 Labelme JSON 中；不要只给整张图片指定一个缺陷类型。**保留包含这些区域类别的原始 JSON**，以后训练缺陷类型模型或分别评估各类型时仍需要它们。

多类型图片可以放入 `defect/mixed/`，对应 mask 放入 `masks/mixed/`；`mixed` 只是整理目录，不代表新增一种缺陷类别。实际类型仍记录在 JSON 的每个区域标签中。单一类型子目录也只是便于整理和采样，目录名不会自动成为当前模型的类型标签。

当前 AA-CLIP 训练使用所有缺陷区域的**二值并集 mask**。已有 Labelme 多边形标注可这样转换，不需要安装 Labelme 才能执行转换：

```bash
.venv/bin/python create_masks.py \
  --annotations data/wafer_dataset/wafer/defect \
  --output data/wafer_dataset/wafer/masks \
  --labels scratch particle contamination
```

`create_masks.py` 读取 JSON，将这三种标签的所有区域合并为一张 `0/255` mask，原始 JSON 保持不变。输出 mask 中不会保留类型编号；预览叠加图位于输出目录的 `_previews/`。如果已经统一标为 `defect`，也可以继续使用 `--labels defect`；新增标签时，将所有使用的标签列入 `--labels`，否则脚本会报未知标签错误。

无论原始区域有多少种类型，训练清单仍填写 `class_name="wafer"`，正常图 `label=0`，含任意缺陷的图 `label=1`。当前模型输出正常 / 异常与异常位置框，**不输出带 `scratch`、`particle`、`contamination` 类型的框**。若以后要识别框内具体缺陷类型，需要使用保留的区域类别标注，修改模型与训练、推理流程；仅修改 `class_name`、目录名或这条 mask 转换命令不会获得类型识别能力。

若用 SAM 辅助标注，先人工确认、修正候选，再导出二值 mask；验证 / 测试标注也应独立复核。

本项目的 `shot=8` 指**每个产品类别共 8 张训练图：4 张正常 + 4 张异常**，不是每种缺陷各 8 张。默认验证集为 4 张正常 + 4 张异常，剩余图片进入测试集。因此脚本运行的数量下限是正常至少 9 张、异常至少 9 张；这个下限只保证三个集合非空，无法证明部署效果。

先用 8-shot 验证流程，再按实际缺陷多样性比较 16-shot、32-shot。4 张异常通常覆盖不了全部缺陷类型、尺寸、位置和成像变化。少样本约束训练样本数；验证和测试应保留尽可能多的独立、具有代表性的样本，并分别统计罕见 / 微小缺陷和正常反光等困难情况。

## 4. 划分：防止同片、同批次泄漏

对于相互独立的图片，可直接生成清单：

```bash
.venv/bin/python split_dataset.py \
  --data_root data/wafer_dataset \
  --classes wafer \
  --shot 8 \
  --val_per_label 4 \
  --seed 111 \
  --output dataset/metadata/wafer_8shot
```

输出 `train.jsonl`、`val.jsonl`、`test.jsonl`、`split_config.json`、`audit.json`，不会移动原图。脚本校验图片与 mask、缺陷是否非空、尺寸是否一致，并通过解码后的像素哈希拒绝完全重复图片。按缺陷子目录交替采样有助于覆盖类型，但不会保证每种类型都出现在三个集合。

**当前划分脚本按图片抽样，不会自动识别 batch / lot / wafer ID。** 同一 wafer 的多视图、同一原图裁出来的 patch、连续视频帧应全部进入同一集合；若要评估跨生产批次泛化，也应以 lot / 批次为组划分。先按组确定 train / val / test，再分别挑训练 few-shot，并手写对应 JSONL。不能把同一片的近似图片随机拆进三个集合；像素哈希无法发现这类泄漏。

手写 JSONL 时每行一个对象，所有路径相对于 `--data_root`，例如：

```jsonl
{"image_path":"wafer/good/normal_001.png","label":0,"class_name":"wafer","mask_path":""}
{"image_path":"wafer/defect/scratch/defect_001.png","label":1,"class_name":"wafer","mask_path":"wafer/masks/scratch/defect_001_mask.png"}
```

训练入口会在加载模型前检查每行数据及 mask。`class_name` 必须是非空产品名称，`label` 必须为 `0` 或 `1`。显式指定 `--meta_path` 后，**实际样本数量由清单决定，`--shot` 不会再次抽样**。开始实验后固定清单，其他随机种子或 shot 使用新目录。

## 5. 两阶段训练

下面是少显存、先调通的起点。第一阶段训练文本适配器；第二阶段训练视觉适配器。CLIP 主干冻结，异常 mask 提供定位监督，图像标签提供正常 / 异常监督。

```bash
.venv/bin/python train.py \
  --dataset Custom \
  --device auto \
  --data_root data/wafer_dataset \
  --meta_path dataset/metadata/wafer_8shot/train.jsonl \
  --shot 8 \
  --save_path ckpt/wafer_8shot \
  --img_size 224 \
  --text_batch_size 1 \
  --image_batch_size 1 \
  --text_epoch 1 \
  --image_epoch 4 \
  --num_workers 0 \
  --grad_checkpointing
```

`224` 用于降低显存占用，不代表 wafer 的最佳分辨率。可在新的模型目录尝试 `336` 或 `518`；分辨率必须是 `14` 的倍数且不少于 `56`。提高分辨率前检查缺陷在缩放后是否仍有可见像素，并评估显存。

模型目录保存 `config.json`、`text_adapter.pth`、`image_adapter.pth` 及按轮次保存的视觉适配器。相同目录已有权重时会续训；`--image_epoch` 是目标总轮数。换训练数据、分辨率或适配器设置时使用新的 `--save_path`，之后重新校准阈值。

新训练默认使用与 OpenAI CLIP 权重一致的 QuickGELU，并将该设置写入配置。旧配置没有 `quick_gelu` 时，续训和推理保留旧实现的 GELU 行为以兼容已有适配器；无需手工改旧配置。对比新旧实验时，使用不同模型目录并分别评估。

当前训练增强包含旋转、平移、翻转和颜色扰动。对于依赖方向的缺陷或严格固定的成像，应检查这些增强是否符合业务含义，再评估调整；不能用测试集决定参数。

## 6. 用验证集校准分类阈值与画框阈值

```bash
.venv/bin/python calibrate_threshold.py \
  --save_path ckpt/wafer_8shot \
  --data_root data/wafer_dataset \
  --meta_path dataset/metadata/wafer_8shot/val.jsonl \
  --pixel_method dice
```

输出模型目录中的 `thresholds.json`，分别保存分类阈值与像素画框阈值：

- **图像分类**：默认使用原始 `pixel_max` 分数，根据验证集正常 / 异常标签选择 Youden J 最大的阈值，即验证集平衡准确率最大；也可用 `--score_type image_score` 重新校准。
- **像素定位**：默认 `--pixel_method dice`，将原始响应还原到原图尺寸，与验证集人工缺陷 mask 比较，选择汇总像素 Dice 最大的阈值。正常图作为全零 mask 参与统计；大图贡献更多像素。这个目标在形态学处理、面积过滤与图像门控之前计算，不是检测 mAP，也不保证框 IoU 或每个缺陷的召回率。

两种阈值都**只使用 val，不使用 test**。人工 mask 质量会影响 Dice 阈值；阈值确定后，还要在验证集检查漏检、误检和画框效果，再固定配置进行独立测试。

若希望按正常验证像素的误报比例校准，可选另一种方法：

```bash
.venv/bin/python calibrate_threshold.py \
  --save_path ckpt/wafer_8shot \
  --data_root data/wafer_dataset \
  --meta_path dataset/metadata/wafer_8shot/val.jsonl \
  --pixel_method normal_fpr \
  --pixel_fpr 0.001
```

`--pixel_fpr` 只在 `normal_fpr` 方法中使用，表示区域后处理之前正常验证像素的目标误报比例，**不是部署中每片 wafer 的误报概率**，也不是漏检率保证。

响应分数允许为负，既不是缺陷概率，也不存在通用的 `0.5` 阈值。热力图显示会按每张图片的范围拉伸，正常图也可能出现红色；画框使用原始响应阈值，不能按热力图颜色判断缺陷。变更模型、分辨率或输入预处理后，应重新校准。

## 7. 预测、画框与结果格式

单图或目录都用同一入口：

```bash
.venv/bin/python custom_infer.py \
  --save_path ckpt/wafer_8shot \
  --class_name wafer \
  --image data/wafer_dataset/wafer/defect/scratch/defect_001.png \
  --output results/wafer_8shot

.venv/bin/python custom_infer.py \
  --save_path ckpt/wafer_8shot \
  --class_name wafer \
  --image data/wafer_incoming \
  --output results/wafer_incoming
```

画框流程为：**异常响应 → 原图尺寸 → 验证集像素阈值 → 二值 mask → 连通区域 → 外接矩形**。它是由异常分割得到的粗定位框，不是训练了物体类别、实例框回归的 YOLO 一类原生检测模型。图像块响应及插值会使边界比人工缺陷轮廓粗；一个裂纹断成几个不连通区域时可能出现多个框，互相连接的多个缺陷可能合成一个框。

输出包含：

| 文件 / 字段 | 含义 |
| --- | --- |
| `*_boxes.png` | 原图上绘制异常区域外接矩形 |
| `*_mask.png` | 有像素阈值时输出的二值异常区域 mask |
| `*_overlay.png`、`*_heatmap.png` | 异常叠加图及可视化热力图 |
| `*_raw.npy` | 原始异常响应，可用于进一步分析 |
| `predictions.json` | 图像分数、正常 / 异常判定和框坐标等结果 |
| `bbox_xyxy` | `[x1, y1, x2, y2]`，原图像素坐标，右下边界为 exclusive；宽高为 `x2-x1`、`y2-y1` |

没有像素阈值时，JSON 中 `detections=null`、`detection_status="uncalibrated"`，不会输出 `*_mask.png`；先校准或指定 `--pixel_threshold` 才能生成检测区域。默认启用图像判定门控：图像分类为 `normal` 时清空检测框和 mask，状态为 `suppressed_by_image_threshold`。若要检查像素定位本身，可以添加 `--no_image_gate` 关闭该门控；应根据验证集漏检 / 误报选择部署策略。

默认最小区域面积 `--min_area 25` 按**原图像素面积**过滤。微小缺陷可能需要降低该值；零碎噪点较多则需要提高。`--close_kernel 0` 表示不做闭运算；需要连接破碎裂纹时可以试奇数核，如 `3`，但这也可能把邻近缺陷合并。参数依据验证集选择并固定，测试集不参与调整。

也可人工指定已经在验证集选定的像素阈值：

```bash
.venv/bin/python custom_infer.py \
  --save_path ckpt/wafer_8shot \
  --class_name wafer \
  --image data/wafer_incoming \
  --output results/wafer_manual \
  --pixel_threshold 0.0 \
  --min_area 25 \
  --close_kernel 0
```

上面的 `0.0` 只是参数格式示例，必须替换为自己验证集确定的阈值。图像分类的手动阈值是另一个参数 `--threshold`，通过 `--score_type pixel_max` 或 `image_score` 指定对应分数；它与像素画框阈值分别使用。

网页上传和预览：

```bash
.venv/bin/python web_infer_app.py --save_path ckpt/wafer_8shot --class_name wafer
```

## 8. 独立评估与部署判断

```bash
.venv/bin/python test.py \
  --dataset Custom \
  --data_root data/wafer_dataset \
  --meta_path dataset/metadata/wafer_8shot/test.jsonl \
  --save_path ckpt/wafer_8shot \
  --batch_size 1 \
  --num_workers 0 \
  --visualize

.venv/bin/python -m unittest discover -s tests -v
```

`test.py` 输出图像 / 像素级 AUROC、AP 等指标及可视化。现有论文式测试评分和部署使用的原始阈值评分有所区别，不能把 AUROC 当成固定阈值准确率。部署还应在独立测试集记录异常漏检率、正常误报率，并按缺陷类型、大小、位置和生产批次分别复核。

画出了框并不证明框正确。若需衡量检测质量，可从人工缺陷 mask 生成真实区域框，按业务规则计算框 IoU、召回和误报；需先明确“一个缺陷实例”的合并规则。当前 `test.py` 的分割 / 分类指标不是检测 mAP。训练少样本数量、阈值和后处理都应只由 train / val 决定，test 留到配置固定后评估。

这个流程可以完成 wafer 正常 / 异常判断及缺陷位置画框；若后续需要输出具体缺陷类别，需要额外的缺陷类型标签与分类或实例检测训练方案，不能通过把 `class_name` 改成缺陷名称自动获得。

## 9. 本次验证记录

- 本机环境：macOS ARM64、Python 3.12.14、PyTorch 2.14.1、torchvision 0.29.1，使用 Apple MPS。依赖安装在 `.venv/`，版本记录在 `requirements-lock-macos.txt`；`pip check` 通过。
- 63 项回归测试通过，覆盖数据与 mask 校验、两阶段训练反传、注意力批次独立性、阈值校准与防失配、框坐标、正常图门控、CLI 和网页。
- 真实 `ViT-L-14-336px.pt` 主干在 224 分辨率完成 8 张 bottle 图片的文本 1 轮、视觉 1 轮训练；权重、配置、训练日志、阈值和 71 张独立测试图片的指标在 `ckpt/bottle_runtime_smoke/`。这一轮仅用于程序流程验证，未针对效果调优。
- 既有 `ckpt/bottle_8shot/` 模型已重新用原验证集校准分类与 Dice 像素阈值，目录预测样例在 `results/wafer_workflow_check/`，网页真实上传也已返回框图、mask 与 JSON。这些均为 bottle 演示，目录名称不表示训练过 wafer。
- CPU 与 MPS 训练回归通过；本机没有 CUDA，未实测 NVIDIA GPU。MPS 的部分反向传播算子不支持严格确定性，训练会明确记录警告并继续；固定随机种子不保证逐位相同。需要严格确定性校验可用 `--device cpu`。

原图缩放、CLIP patch 特征以及阈值都会影响定位，外接框可能包含正常像素。细小 wafer 缺陷的分辨率、误报、漏检和框范围，应使用自己的独立验证与测试数据确认。
