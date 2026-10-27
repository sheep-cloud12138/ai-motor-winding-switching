# WIP 审计状态（2026-10-27）

## 已确认并保存

- 所有正在运行的 Python/FEMM 审计进程已停止。
- 原先“转矩漏乘 L_STACK”的结论**未通过能量一致性检验**，不能当作最终修复。
- 自研求解器数据已恢复到历史原始转矩尺度，驱动模型检查点中的转矩标准化参数同步恢复。
- 新增统一三方检验：
  - 同一 `withz` MLP；
  - 代理 vs 自研求解器；
  - 自研求解器 vs FEMM；
  - 代理 vs FEMM。
- 修正了未见连接相轴错误。旧 tap50 检验错误地使用了驱动循环相轴，与基准训练/测试相轴不同。
- 修正后，3S1P 在当前三方检验中表现最好；4P、tap50 的转矩原始 R² 仍低。
- 三种连接现在均使用同一个 `withz` MLP：
  - 3S1P：代理 vs 自研转矩相关 0.818、R² 0.594；自研 vs FEMM 0.989、R² 0.952。
  - 4P：代理 vs 自研 0.604、R² -0.194；自研 vs FEMM 0.935、R² 0.858。
  - tap50：代理 vs 自研 0.735、R² -0.028；自研 vs FEMM 0.945、R² 0.877。

## 关键未解决问题

1. 转矩标签生成已找到正确修复路径，但尚未全量重建。
   - 旧的槽内 Lorentz 转矩与磁链不满足离散 Maxwell 关系。
   - 新增 `exp/energy_torque.py`：从离散磁静能量泛函对转子角求导。
   - 40×96 网格、4S、40 A 验证：
     - `dT/di` 与 `-dψ/dθ` 的相关性为 **0.9994**；
     - 相对范数残差为 **4.0%**。
   - 符号约定为 `∂T/∂i = -∂ψ/∂θ`。
   - 还需要用该函数重建正式数据集、检查点、FEMM对照和驱动循环。
2. PM基波仍差约 41.8%，电感仍有约 1.13 倍绝对归一化差。
3. `results_drive.json` 与当前单电机驱动配置尚未重新对齐；中断前未完成全量重跑。
4. 没有实测或 HIL 数据。

## 不应再使用的结果

- 旧 tap50“转矩相关 -0.21”：连接矩阵错误，已作废。
- 由转矩乘/除 10 驱动的车辆规模与 15%/41% 结论：未通过 Maxwell 关系，全部暂缓引用。

## 恢复工作顺序

1. 固定转矩与磁链的符号和量纲，使 `d(psi)/d(theta) = d(T)/di`。
2. 重新生成受影响数据与检查点。
3. 重跑 FEMM 三方检验和驱动循环。
4. 更新论文表格、摘要和结论后再提交到主分支。

## 参考文件

- `exp/results_energy_consistency.json`
- `exp/energy_torque.py`
- `exp/validate_energy_torque.py`
- `exp/results_energy_torque_validation.json`
- `exp/review_energy_modes.py`
- `exp/review_slot_relation.py`
- `exp/results_unseen_consistency.json`
- `exp/results_4p_femm_withz.json`
- `exp/results_tap50_femm_withz.json`
- `exp/results_3s1p_femm.json`
- `exp/femm_unseen_withz.py`
- `exp/femm_drive_candidate.py`
