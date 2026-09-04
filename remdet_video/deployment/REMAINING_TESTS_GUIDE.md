# RemDet Jetson 剩余测试一键操作清单

这份清单用于下次 Jetson 在身边时一次性完成剩余实验，不需要逐步等待确认。

## 将要自动完成的内容

1. 检查 FP16 engine、脚本、视频和 548 张 VisDrone 验证图片。
2. 对 720p/30 FPS/H.264 视频执行 3 轮完整流水线测试。
3. 自动尝试 OpenCV 普通解码和 Jetson GStreamer/NVDEC 硬件解码。
4. 对全部 548 张 VisDrone 验证图片执行 TensorRT FP16 推理。
5. 记录每个环节的平均延迟、P95、FPS、功耗、温度和内存。
6. 保存 COCO 格式预测结果并将 Jetson 上的所有结果打包。
7. 回传电脑后自动计算 mAP，并与电脑 PyTorch 基准比较。

## 第一步：上传一个测试包（Windows PowerShell）

```powershell
scp "C:\Users\xh\Desktop\WindyLab\RemDet\work_dirs\deployment\remdet_remaining_tests_bundle.tar.gz" nvidia@192.168.55.1:/home/nvidia/remdet_deploy/
```

出现密码提示时输入 Jetson 用户 `nvidia` 的密码。

测试包大小约 78.2 MiB，SHA256 为：

```text
25e82516d006c59ae4b962e528cd0f4d6f354e7ea13a937943f7d4f19365b7b6
```

正常情况下不需要单独停下来校验；只有上传报错或解压失败时再检查该哈希。

## 第二步：登录 Jetson（Windows PowerShell）

```powershell
ssh nvidia@192.168.55.1
```

## 第三步：一键解压并执行（Jetson SSH 终端）

```bash
tar -xzf /home/nvidia/remdet_deploy/remdet_remaining_tests_bundle.tar.gz -C /home/nvidia/remdet_deploy && bash /home/nvidia/remdet_deploy/scripts/run_remaining_tests.sh
```

测试完成时应看到：

```text
All remaining Jetson tests completed successfully.
Result bundle: /home/nvidia/remdet_deploy/remdet_all_results.tar.gz
```

即使某个测试报错，脚本也会尽量把已有日志打包到同一个结果文件中。

## 可选：让测试在后台运行

如果不想保持 SSH 窗口，可以先解压：

```bash
tar -xzf /home/nvidia/remdet_deploy/remdet_remaining_tests_bundle.tar.gz -C /home/nvidia/remdet_deploy
```

然后后台启动：

```bash
nohup bash /home/nvidia/remdet_deploy/scripts/run_remaining_tests.sh >/home/nvidia/remdet_deploy/results/nohup_launcher.log 2>&1 &
```

查看进度：

```bash
tail -f /home/nvidia/remdet_deploy/results/remaining_tests_console.log
```

看到测试完成后按 `Ctrl+C` 只是退出日志查看，不会删除测试结果。

## 第四步：一键下载并计算 mAP（Windows PowerShell）

回到 RemDet 项目目录：

```powershell
cd C:\Users\xh\Desktop\WindyLab\RemDet
```

运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\remdet_video\deployment\collect_jetson_results.ps1
```

脚本会执行一次 `scp`，因此仍需输入一次 Jetson 密码。随后会自动：

- 解压全部结果；
- 使用 `remdet5080` 环境和 pycocotools 计算 COCO mAP；
- 与电脑端 RemDet-S 640 PyTorch 基准比较；
- 生成最终精度 JSON。
- 生成对比 CSV、Markdown 部署报告和汇报用 PNG 图表。

## 电脑端比较基准

| 指标 | PyTorch基准 |
|---|---:|
| bbox mAP | 0.247 |
| AP50 | 0.415 |
| AP75 | 0.250 |
| AP-small | 0.154 |
| AP-medium | 0.367 |
| AP-large | 0.470 |

默认允许每项最多 `0.002` 的绝对差异。超出时不会删除数据，只会将最终状态标为未通过，供进一步检查。

## 最终结果位置

电脑端目录：

```text
C:\Users\xh\Desktop\WindyLab\RemDet\work_dirs\deployment\jetson_results
```

重要文件：

```text
video_pipeline_fp16_15w.json
demo_720p_fp16_detected.mp4
visdrone_val_inference_trt_fp16_15w.json
visdrone_val_predictions_trt_fp16.json
visdrone_val_coco_eval.json
jetson_deployment_summary.json
jetson_performance_comparison.csv
jetson_deployment_report.md
jetson_deployment_comparison.png
environment_inventory.txt
remaining_tests_console.log
```

## 注意事项

- 整个过程只推理，不训练模型。
- 不会修改 FP16 engine。
- 数据集约 83 MiB，Jetson 当前存储空间足够。
- 测试时不要同时运行其他 GPU 程序，否则性能数据会受到干扰。
- 视频实验验证部署功能和速度，不代表已经完成“黄色面包车”专用错漏检实验。
