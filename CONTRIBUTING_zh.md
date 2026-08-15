# 参与 InferTrain 开发贡献指南

感谢您对 InferTrain 的关注与贡献！无论您是报告 Bug、提出新特性，还是提交代码，本指南都能帮助您快速上手。

在开始贡献之前，请先阅读[贡献者许可协议 (CLA)](CLA_zh.md)（中文版仅供参考，以[英文版 CLA.md](CLA.md) 为准）与 [Apache-2.0 许可证](LICENSE)。

## 目录

- [贡献者许可协议 (CLA)](#贡献者许可协议-cla)
- [开发优先级](#开发优先级)
- [开发文档](#开发文档)
- [贡献流程](#贡献流程)
- [编码规范](#编码规范)
- [提交规范](#提交规范)
- [Pull Request 检查清单](#pull-request-检查清单)
- [报告问题](#报告问题)
- [联系方式](#联系方式)

## 贡献者许可协议 (CLA)

所有贡献者都必须在贡献被合并之前签署[贡献者许可协议](CLA_zh.md)。当您第一次打开 Pull Request 时，[cla-assistant](https://cla-assistant.io) 会自动在 PR 下评论并附上签署链接，同时添加 CLA 状态检查——签署过程只需一分钟。

CLA 用于明确您的贡献所授予的知识产权，与 Apache-2.0 许可证完全一致。签署后您并不会失去对自己代码的所有权，只是授予项目一份永久、不可撤销的许可，允许在 Apache-2.0 条款下分发您的贡献。

## 开发优先级

参与开发请优先参考 [TODO_zh.md](TODO_zh.md)。

## 开发文档

- 新算子开发请参考 [docs/dev_ops_zh.md](./docs/dev_ops_zh.md)
- 算子融合优化开发请参考 [docs/dev_fusion_zh.md](./docs/dev_fusion_zh.md)
- 硬件平台接入请参考 [docs/dev_platform_zh.md](./docs/dev_platform_zh.md)
- **写代码之前请先阅读** [docs/rust_subset_guidelines_zh.md](./docs/rust_subset_guidelines_zh.md)

## 贡献流程

1. Fork 仓库
2. 创建功能分支
3. 实现功能
4. 测试您的改动
5. 提交您的改动（参见[提交规范](#提交规范)）
6. 提交 Pull Request

## 编码规范

- 确保代码和注释中无中文
- 确保代码**严格不超过 80 列**，包括注释

## 提交规范

- 保证每次 commit 都是一个完整的、独立的、小型的改动
- 确保 commit msg 无中文
- commit 时加 `-s` 参数，以便生成 Sign-off：

```bash
git commit -s -m "您的提交信息"
```

## Pull Request 检查清单

提交 Pull Request 之前，请确认：

- [ ] 已签署贡献者许可协议（首次 PR 必须）
- [ ] 代码和注释中无中文
- [ ] 代码严格不超过 80 列，包括注释
- [ ] 每次 commit 都是完整的、独立的、小型的改动
- [ ] commit msg 无中文，并使用 `-s` 生成 Sign-off

## 报告问题

- 先搜索已有 issue，避免重复
- 使用清晰且有描述性的标题
- 提供复现步骤、预期行为与实际行为
- 附上环境信息（操作系统、架构、Rust 版本等）

## 联系方式

- Issues: GitHub Issues
- 讨论: GitHub Discussions
