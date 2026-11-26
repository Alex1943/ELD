# ELD 单帧 RAW 对照适配

新增 `train_raw.py` 和 `dataset/raw_manifest.py`，复用 ELD 原始 `UNetSeeInDark` 四通道 U-Net 与 Adam，不改原 SID/LMDB 入口、模型定义或现有依赖。当前适配入口使用 Bracketing 风格 μ-law RAW L1（预测/16仅clamp负值，GT/16 clamp到0..1，μ=5000），与 ELD 原始线性像素 L1 不同；修正文档此前错误描述，本次路径修复未改损失。任务输入输出均为单帧 `B×4×H×W` packed Bayer `[R, Gr, Gb, B]`；输入 uint16 除以1023，GT作为线性 HDR 浮点直接读取。训练只随机裁 patch，不对 Bayer 通道或相位做增强。

数据通过 DatasetGen `manifest.csv` 读取，训练严格选 `split=train`，测试严格选 `split=test`，支持 `subsets` 过滤。正式指标动态调用 DatasetGen `degradation_metrics.RawMetricProtocol`（RAW/16、metadata-aware ISP、learned demosaic、μ=5000），逐样本输出四项指标。每轮训练按总 iteration 等分为8个保存节点，每个节点之后立即完整测试；图片按测试集全局样本序号以 `save_image_interval` 抽样，0表示不保存。输出含模型、每节点 CSV、子集/overall JSON 和可选 PNG。

配置在 `options/raw_eld.json`。`data_root` 和 `datasetgen_root` 各含 `windows`/`linux` 两项，入口按运行平台自动选择，不需要跨平台手动改配置。Windows 默认数据根为 `D:/work/pycharm_work/DatasetGen/hdr_video_data/result_single_recipe`，DatasetGen 根为 `D:/work/pycharm_work/DatasetGen`。OneRestore v2.3 正式服务器配置核实的路径为：数据根 `/mnt/0c2bda26-6d90-4191-a48d-8f7367e86893/students/lihaotian/DataSet/raw_dege_1.3_gtAdjust`；DatasetGen/ISP 根 `/mnt/0c2bda26-6d90-4191-a48d-8f7367e86893/students/lihaotian/raw-single/raw-data-gen`；learned demosaic 权重为该根下的 `isp/model.bin`。启动时打印两项生效根和权重路径，并检查manifest、指标模块和权重是否存在；Linux误用Windows盘符时直接报错。Windows小样本集可能没有test，不能视作正式服务器测试集。GPU 可通过 `--gpu-ids 0` 或 `--gpu-ids 2,3` 选物理卡；多卡使用 DataParallel，batch 是每卡配置乘卡数。

```bash
conda activate seesr
python train_raw.py --config options/raw_eld.json --check-paths
python train_raw.py --config options/raw_eld.json --gpu-ids 0
python train_raw.py --config options/raw_eld.json --gpu-ids 0,1
```

2026-11-26修复：原配置误把Windows路径用于Linux，现由入口自动选择平台并预检；直接按文件加载ELD原U-Net，避免`models/__init__.py`提前导入SID/rawpy及ELD的`util`，污染DatasetGen同名模块；损失tone函数直接来自DatasetGen协议。GPU可见性在导入torch前设置。后续针对服务器漏同步`raw_paths.py`导致的启动失败，将路径处理内联到`train_raw.py`和`dataset/raw_manifest.py`，两者均不再依赖该辅助文件。同步这两个文件及`options/raw_eld.json`即可启动。路径回归检查：`python -B -m unittest discover -s tests -v`。未安装、升级或修改已有依赖；服务器真实数据与CUDA训练仍需同步后验证。

验证：6项路径回归与语法检查通过；本机`conda run -n seesr`下原U-Net 32×32四通道前向/反向、DatasetGen模块来源检查和真实`isp/model.bin`自比较评测通过（SSIM/μ-SSIM=1）。未在Linux服务器访问真实数据或运行完整训练。

2026-11-26测试耗时修复：旧入口在评测前把预测`.cpu()`，GT/metadata也在CPU，导致每图两次learned ISP在CPU执行；现预测、GT、metadata随网络设备在GPU执行ISP，仅量化RGB与skimage PSNR/SSIM在CPU。测试改用batch1 DataLoader，worker默认沿用`workers`，可选`test_workers`覆盖；完整测试仍逐样本计算全部四项指标。单图评测直接使用DataParallel底层模型，避免多卡包装开销；评测结束或异常恢复先前训练模式。同步计时输出读取等待、传输/网络、ISP/指标、保存耗时及`test_XX_timing.json`，实际服务器提速待复测。训练iter日志包含累计elapsed（含已发生的测试）、区间samples/s和ETA：首次测试前ETA仅估剩余训练，之后加上已测平均测试耗时乘剩余测试次数；每次保存/测试后重置速度窗口，避免测试耗时污染训练速度。精确生成8个保存节点（非整除总步数也无额外保存）；可选PNG按RGB保存，修正旧PIL输出误反转R/B。网络、损失、测试选择和DatasetGen指标参数不变。

验证：seesr环境10项回归通过（含CPU/CUDA设备、padding、异常后恢复模式）；另用临时合成RAW数据完成10步真实GPU训练、8次保存及8次真实learned GPU ISP评测，PNG和耗时JSON生成成功。临时输出未进入项目；未据32×32小图冒烟推算服务器全尺寸耗时。

训练记录：控制台与`<output_dir>/<name>/train.log`同步记录启动命令、experiment_note、生效路径、batch/worker/设备、iter/elapsed/ETA、保存节点、每轮测试的分退化及overall表（count、PSNR、SSIM、μ-PSNR、μ-SSIM）和耗时。overall按样本均值，CSV/JSON保留完整精度。进度条仅显示在控制台，不向日志追加逐样本输出；异常堆栈写入train.log。同目录保存源配置原件`source_config.json`（若传入YAML则保留对应扩展名）、统一YAML格式`source_config.yaml`和`resolved_config.yaml`；后者记录平台解析路径、CLI覆盖GPU、global_batch_size及test_workers。入口同时支持JSON/YAML，原JSON启动命令继续可用；YAML读取使用环境已有PyYAML。重复同name启动日志追加、已有配置快照保留，新快照附北京时间戳；不表示支持resume或避免覆盖同名checkpoint，新实验仍需独立name。本次修改不改变网络、损失或指标协议。

记录验证：11项回归通过，覆盖控制台/文件同步、原配置字节保留、CLI覆盖快照和重复启动快照保留；YAML入口另用临时数据完成真实GPU训练及两退化样本learned ISP评测，分退化/overall表和迭代ETA完整写入train.log。使用高精度单调perf_counter修正Windows极短评测零耗时问题。
