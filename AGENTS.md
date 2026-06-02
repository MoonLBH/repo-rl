# 仓库说明

本仓库是一个迁移工作仓库，不是最终发布仓库。

目录结构：
- sotmol-rl/：我的方法代码，暂时命令为LIFT。该方法使用 RL / reward-guided fine-tuning 调整无条件 3D flow matching 分子生成模型。train——lift_*.py文件是我的方法在不同任务上的训练脚本。
- flowr/：目标代码库。该项目是 structure-based 的三维配体生成与优化框架。
- migration_notes/：只用于保存迁移分析文档、接口映射和实施计划。

工作原则：
1. 在没有明确要求前，不要修改 sotmol-rl/ 和 flowr/ 的代码。
2. 先分析，再计划，最后才实现。
3. sotmol-rl/ 是算法来源，flowr/ 是迁移目标。
4. 不要机械复制 sotmol-rl/ 的代码到 flowr/。
5. 需要抽象出 sotmol-rl/ 的计算逻辑，再适配 flowr/ 的数据结构、模型接口和训练流程。
6. 所有新增功能必须默认关闭，不能破坏 flowr/ 原始训练、采样和评估流程。
7. 每个阶段都必须提供可运行的 smoke test 或最小验证命令。
8. 所有迁移说明文档放在 migration_notes/ 下。
