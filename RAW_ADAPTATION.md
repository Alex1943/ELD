# ELD 单帧 RAW 对照适配

新增 `train_raw.py` 和 `dataset/raw_manifest.py`，复用 ELD 原始 `UNetSeeInDark` 四通道 U-Net、Adam 与像素 L1，不改原 SID/LMDB 入口、模型定义或现有依赖。任务输入输出均为单帧 `B×4×H×W` packed Bayer `[R, Gr, Gb, B]`；输入 uint16 除以1023，GT作为线性 HDR 浮点直接读取。训练只随机裁 patch，不对 Bayer 通道或相位做增强。

数据通过 DatasetGen `manifest.csv` 读取，训练严格选 `split=train`，测试严格选 `split=test`，支持 `subsets` 过滤。正式指标动态调用 DatasetGen `degradation_metrics.RawMetricProtocol`（RAW/16、metadata-aware ISP、learned demosaic、μ=5000），逐样本输出四项指标。每轮训练按总 iteration 等分为8个保存节点，每个节点之后立即完整测试；图片按测试集全局样本序号以 `save_image_interval` 抽样，0表示不保存。输出含模型、每节点 CSV、子集/overall JSON 和可选 PNG。

配置在 `options/raw_eld.json`。Windows 默认数据根为 `D:/work/pycharm_work/DatasetGen/hdr_video_data/result_single_recipe`，DatasetGen 根为 `D:/work/pycharm_work/DatasetGen`。OneRestore v2.3 正式服务器配置核实的路径为：数据根 `/mnt/0c2bda26-6d90-4191-a48d-8f7367e86893/students/lihaotian/DataSet/raw_dege_1.3_gtAdjust`；DatasetGen/ISP 根 `/mnt/0c2bda26-6d90-4191-a48d-8f7367e86893/students/lihaotian/raw-single/raw-data-gen`；learned demosaic 权重 `/mnt/0c2bda26-6d90-4191-a48d-8f7367e86893/students/lihaotian/raw-single/raw-data-gen/isp/model.bin`。在服务器运行 ELD 时，将配置中的 `data_root` 和 `datasetgen_root` 分别改为对应服务器路径，确保代码与 `isp/model.bin` 同步。GPU 可通过 `--gpu-ids 0` 或 `--gpu-ids 2,3` 选物理卡；多卡使用 DataParallel，batch 是每卡配置乘卡数。

```bash
conda activate seesr
python train_raw.py --config options/raw_eld.json --gpu-ids 0
python train_raw.py --config options/raw_eld.json --gpu-ids 0,1
```

本次只新增 ELD 项目内适配文件，没有安装、升级或修改已有依赖。当前仅完成 Python 语法检查；完整数据读取、CUDA 训练和 DatasetGen learned ISP 测试需在 `seesr` 环境与有数据/ISP 权重的机器上验证。
