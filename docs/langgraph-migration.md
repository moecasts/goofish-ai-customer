# LangGraph 迁移指南

本文档说明如何从旧版 Agent 系统迁移到 LangGraph 架构。

## 架构对比

### 旧版架构
```
用户消息 → 关键词匹配 → LLM分类 → Agent分发 → 生成回复
```

### 新版架构（LangGraph）
```
用户消息 → 意图识别节点 → 条件路由 → 议价/商品/默认节点 → 输出
```

## 迁移步骤

### 1. 启用 LangGraph
在 `.env` 文件中设置：
```bash
USE_LANGGRAPH=true
```

### 2. 验证功能
启动服务并测试：
```bash
python main.py
```

### 3. 回滚方案
如果遇到问题，可以立即回滚：
```bash
USE_LANGGRAPH=false
```

## 功能差异

### 新增功能
1. **会话隔离** - 每个用户独立的状态
2. **议价循环** - 支持多轮议价
3. **监控支持** - 可选的 LangSmith 集成

### 保持兼容
- 所有现有 prompt 文件不变
- API 接口保持兼容
- 配置文件格式不变
