# 更新日志

## [2025-09-05] - API 迁移更新

### 🔄 重大变更
- **API 提供商迁移**: 将所有 Cohere API 调用替换为 Jina AI API
- **嵌入模型更新**: 使用 `jina-embeddings-v4` 模型，支持 2048 维嵌入向量

### ✨ 新功能
- 支持 Jina AI 的多模态嵌入生成（文本 + 图像）
- 保持原有的图文对齐和去重功能
- 优化的 API 调用性能和错误处理

### 🔧 技术变更
- 移除 `cohere` 依赖包
- 更新配置项：`COHERE_API_KEY` → `JINA_API_KEY`
- 更新 Agent 框架嵌入器：`CohereEmbedder` → `JinaEmbedder`
- 重构 `MultimodalService` 类使用 HTTP 请求

### 📝 配置更新
需要在环境变量或 `.env` 文件中设置：
```
JINA_API_KEY=your_jina_api_key
```

### 🔗 兼容性
- 保持所有原有接口不变
- 现有功能完全兼容
- 嵌入向量维度从 1024 提升到 2048

---

## 历史版本
更多历史更新请查看 Git 提交记录。
