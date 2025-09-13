# TOSEM 工厂合约分析系统 (Google BigQuery 版)

## 概述

这是一个基于 **Google BigQuery** 的工厂合约检测系统，用于分析 5 个 EVM 区块链网络的合约字节码，识别工厂合约类型。

### 支持的区块链
- **Ethereum** (以太坊主网)
- **Polygon** (Polygon 主网) 
- **Arbitrum** (Arbitrum One)
- **Optimism** (Optimism 主网)
- **Avalanche** (Avalanche C-Chain)

### 系统架构
- **数据源**: Google BigQuery 公开数据集
- **存储**: Google BigQuery 自定义表
- **处理**: 多线程并发分析
- **检测**: 集成 factory_detector.py 模块

## 核心功能

### 🔍 工厂合约检测
- 识别使用 `CREATE` 操作码的工厂合约
- 识别使用 `CREATE2` 操作码的工厂合约  
- 识别同时使用两种操作码的混合工厂合约
- 记录分析时间和成功率

### 📊 大规模数据处理
- 按月分批处理，避免查询超时
- 支持 2025年6月1日前的全部历史合约
- 断点恢复机制，中断后可继续处理
- 并发处理 5 个区块链

### 💾 智能存储
- 自动创建 BigQuery 数据集和表
- 分区和聚集优化查询性能
- 进度跟踪避免重复处理
- 批量插入提高写入效率

## 快速开始

### 1. 环境准备
```bash
# 安装依赖
pip install -r requirements.txt

# 配置 Google Cloud 认证
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
```

### 2. 配置系统
在 `analysis_all_chain_factory.py` 顶部修改配置：
```python
BIGQUERY_CONFIG = {
    "project_id": "your-google-cloud-project-id",  # 必填
    "service_account_key_file": "/path/to/key.json",  # 可选
    # 其他配置已预设
}
```

### 3. 运行分析
```bash
# 测试配置（干运行）
python analysis_all_chain_factory.py  # 设置 dry_run=True

# 正式分析
python analysis_all_chain_factory.py  # 设置 dry_run=False
```

## 文件结构

```
TOSEM/
├── analysis_all_chain_factory.py    # 主程序
├── factory_detector.py              # 工厂合约检测器
├── blockchain_config.json           # 区块链配置
├── requirements.txt                 # Python 依赖
├── BIGQUERY_SETUP_GUIDE.md         # 详细配置指南
├── README.md                        # 本文件
├── bytecode/                        # 字节码相关文件
└── data/                           # 数据目录
```

## 详细配置

### BigQuery 配置选项
```python
BIGQUERY_CONFIG = {
    "project_id": "",                    # Google Cloud 项目 ID
    "dataset_id": "tosem_factory_analysis", # 数据集名称
    "location": "US",                    # 数据集位置
    "batch_size_months": 1,              # 每批处理月数
    "use_cache": True,                   # 启用查询缓存
    "dry_run": False,                    # 干运行模式
}
```

### 区块链配置
- **Ethereum & Polygon**: 直接查询合约表
- **Arbitrum & Optimism & Avalanche**: JOIN 交易表和收据表

### 分析配置
```python
ANALYSIS_CONFIG = {
    "cutoff_date": "2025-06-01",     # 分析截止日期
    "max_workers": 5,                # 最大并发数
    "batch_save_size": 1000,         # 批量保存大小
}
```

## 数据表结构

### factory_analysis_results
主要分析结果表：
- `chain`: 区块链名称
- `address`: 合约地址
- `is_factory`: 是否为工厂合约
- `is_create`: 支持 CREATE
- `is_create2`: 支持 CREATE2  
- `is_both`: 同时支持两者
- `analysis_success`: 分析是否成功
- `analysis_time`: 分析时间（毫秒）
- `processed_at`: 处理时间

### analysis_progress  
进度跟踪表：
- `chain`: 区块链名称
- `start_date/end_date`: 时间段
- `status`: 处理状态
- `contracts_processed`: 处理的合约数
- `factories_found`: 发现的工厂数

## 性能特点

### 🚀 高性能
- 5 线程并发处理
- BigQuery 原生查询优化
- 内存友好的批量处理

### 💰 成本优化
- 查询缓存减少重复计算
- 分区过滤减少扫描数据量
- 免费额度：每月 1TB 查询

### 🛡️ 可靠性
- 断点恢复机制
- 详细错误处理和日志
- 自动重试失败查询

## 使用示例

### 查看分析结果
```sql
-- 查看各链工厂合约统计
SELECT 
  chain,
  COUNT(*) as total_contracts,
  COUNT(CASE WHEN is_factory THEN 1 END) as factory_contracts,
  COUNT(CASE WHEN is_create THEN 1 END) as create_factories,
  COUNT(CASE WHEN is_create2 THEN 1 END) as create2_factories,
  COUNT(CASE WHEN is_both THEN 1 END) as mixed_factories
FROM `your-project.tosem_factory_analysis.factory_analysis_results`
GROUP BY chain
ORDER BY factory_contracts DESC;
```

### 监控处理进度
```sql
-- 查看处理进度
SELECT 
  chain,
  status,
  COUNT(*) as periods,
  SUM(contracts_processed) as total_processed,
  SUM(factories_found) as total_factories
FROM `your-project.tosem_factory_analysis.analysis_progress`  
GROUP BY chain, status
ORDER BY chain, status;
```

## 故障排除

### 常见问题
1. **认证失败**: 检查服务账户密钥和项目ID
2. **权限不足**: 确认 BigQuery 权限配置
3. **配额超限**: 调整并发数和批量大小
4. **查询超时**: 减少时间窗口大小

### 调试技巧
- 启用 `dry_run` 模式验证查询
- 检查日志文件：`bigquery_factory_analysis.log`
- 使用详细日志级别：`logging.DEBUG`

## 扩展性

### 添加新区块链
1. 在 `BLOCKCHAIN_CONFIGS` 中添加配置
2. 确认 BigQuery 数据集可用
3. 根据数据结构选择查询类型

### 自定义分析逻辑
- 修改 `analyze_contract` 方法
- 扩展结果数据结构
- 更新表 schema

## 更多信息

详细的配置和使用指南请参考：
- [BIGQUERY_SETUP_GUIDE.md](BIGQUERY_SETUP_GUIDE.md) - 完整配置指南
- [blockchain_config.json](blockchain_config.json) - 区块链配置详情

---

**注**: 本系统完全基于 Google BigQuery，具有高性能和可扩展性。请确保正确配置认证和权限后使用。
