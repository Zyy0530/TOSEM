# RQ3 工厂合约聚类分析：思考与设计

目标：从 `rq2_factory_creations` 中获取 Ethereum 与 Polygon 两条链的工厂合约地址，尽可能快速地判定哪些工厂合约已被验证并获取其源码（仅使用 Sourcify 命中的源码；未命中直接跳过），构建面向“语义 + 机制 + 产物”的多视角特征，选择合适的聚类方法并绘制二维语义空间图。分析需按链分别进行（Ethereum 与 Polygon 各生成一张图），并保证聚类尽量“正交”（不同维度的因素尽量相互独立）。

---

## 1) 读取 rq2_factory_creations，获取两条链的工厂合约地址

- 表位置：读取 `experiments/RQ2/config.json` 中的 `project_id` 与 `result_dataset`，表名固定为 `rq2_factory_creations`，最终 FQN 形如：
  - `ziyue-wang.tosem_factory_analysis.rq2_factory_creations`
- 建议 SQL（去重、统计基础信息，供后续过滤工厂“强度”）：

```
SELECT
  chain,
  LOWER(factory_address) AS factory_address,
  COUNT(*) AS total_creations,
  COUNT(DISTINCT created_contract_address) AS unique_creations,
  MIN(block_date) AS first_seen_date,
  MAX(block_date) AS last_seen_date
FROM `PROJECT.DATASET.rq2_factory_creations`
WHERE chain IN ('ethereum', 'polygon')
GROUP BY chain, factory_address
HAVING unique_creations >= 1  -- 可调：>=2 可更稳健识别“工厂”
```

- Python 读取（google-cloud-bigquery）：

```
from google.cloud import bigquery
import json, os

cfg = json.load(open('experiments/RQ2/config.json'))
client = bigquery.Client(project=cfg['project_id'])
table = f"{cfg['project_id']}.{cfg['result_dataset']}.rq2_factory_creations"
sql = f"""
SELECT chain, LOWER(factory_address) AS factory_address,
       COUNT(*) AS total_creations,
       COUNT(DISTINCT created_contract_address) AS unique_creations
FROM `{table}`
WHERE chain IN ('ethereum','polygon')
GROUP BY chain, factory_address
"""
df_factories = client.query(sql).result().to_dataframe()
```

- 备注：`rq2_factory_creations` 已包含 `runtime_code`（创建产物的运行时代码）。本次分析仅针对 Sourcify 命中的工厂源码样本，但仍可结合其“产物”的运行时代码作为附加特征（例如 EIP‑1167、Proxy/Beacon、Token/NFT 迹象）。

---

## 2) 根据地址快速筛选“已验证”合约，并快速获取源码

优先级与策略：

- Sourcify（多链、开源、结构化源码包，唯一来源）：
  - 优点：一次性支持多链（Ethereum=1，Polygon=137），可批量查询验证状态（`POST /check-all-by-addresses`）。
  - 步骤：
    - 将 `df_factories` 按链分组，映射 chain→chainId（ethereum→1，polygon→137）。
    - 每链对 `factory_address` 分块（建议每 50–100 个地址一批）调用 `check-all-by-addresses`，确认 `full_match` 或 `partial_match`。
    - 对已验证地址，使用 `GET /files/:chainId/:address` 拉取源码文件树和 `metadata.json`（包含编译器版本、settings 等）。
  - 缓存：落地到本地 `experiments/RQ3/data/sources/{chain}/{address}.json`，并可同步到 BQ 表（建议 `tosem_factory_analysis.factory_sources`）。

- 未命中策略：未在 Sourcify 命中的地址直接跳过（不再使用 Etherscan/Polygonscan）。仅分析 Sourcify 命中的源码样本。

- 建议的 BQ 落表（可选）：`tosem_factory_analysis.factory_sources`
  - 字段：`chain STRING, address STRING, source_status STRING, files ARRAY<STRUCT<path STRING, content STRING>>, compiler STRING, evm_version STRING, verified_at TIMESTAMP, origin STRING`（origin 固定为 `sourcify`）。

---

## 3) 设计聚类方法：算法与特征选择

总体思路：多视角（Mechanism/机制、Semantics/语义、Products/产物）构建特征 → 归一化与正交化 → 无监督聚类（噪声鲁棒）→ 可解释命名与二维可视化。

- 3.1 特征设计（按“工厂自身”和“工厂产物”两个来源）
  - 工厂合约（源码或字节码）机制特征：
    - 部署方式：`has_create`、`has_create2`、是否使用 `Clones`（EIP‑1167 字节序列检测）、是否 `DETERMINISTIC`（存在 `salt` 参数）。
    - 访问控制：是否使用 `Ownable`/`AccessControl`，是否 `onlyOwner` 修饰符，公共/外部入口函数占比。
    - 结构复杂度：事件数量、`mapping`/数组数量、函数总数、代码行数/长度、是否使用 OZ 库。
    - 语义线索（源码时）：`create*`/`deploy*`/`clone*`/`predict*` 命名模式、业务词汇命中（DEX、NFT、DAO、wallet、vault 等）。
  - 工厂产物（由 `rq2_factory_creations.runtime_code` 聚合而来）：
    - EIP‑1167 Minimal Proxy 检测（`0x3d602d80600a3d3981f3...` 等字节序列）。
    - 代理/可升级迹象：是否包含 `DELEGATECALL`、EIP‑1967 插槽常量、Beacon/UUPS 模式特征。
    - 业务接口线索：运行时代码中函数选择子分布（token/NFT/DEX/wallet 常见 selectors）。
    - 产物多样性：一个工厂创建的不同运行时代码的去重计数、主要类别占比。

- 3.2 特征工程
  - 将特征分为三大块：`Mechanism`（机制）、`Access/Control`（权限/结构）、`Product`（产物）。
  - 数值标准化：`StandardScaler`。
  - 降维与正交化：
    - 先对每个特征块做 PCA(保留解释度 80–90%)，再进行 Varimax 旋转提升可解释性与正交性；
    - 或直接做 ICA（独立成分分析）得到近似独立轴。
  - 融合：拼接三个子空间后的向量，并进行整体 whitening（ZCA），确保整体协方差接近单位阵。

— 不去重策略与影响度量 —

- 不进行“源码等价类”去重：每个工厂地址作为独立样本参与特征构建、降维与聚类，以真实反映“占比/影响力”（重复更多的类型在簇中占比更大）。
- 为保证计算效率且不改变统计贡献，可在实现层面“等价点聚合 + sample_weight=频次”（KMeans/HDBSCAN 均支持权重），但输出与评估均按未去重口径进行；若无需优化，则直接对全部样本建模。
- 可视化时使用透明度（alpha）与密度/核估计表现“高频聚集”，或用点大小按 `unique_creations`/`total_creations` 缩放，突出影响力。

- 3.3 聚类算法选择
  - 基准：HDBSCAN（处理不同密度、可自动识别噪声，减少手动选 K），对 outlier 更稳健；
  - 备选：
    - KMeans 或 Spherical KMeans（在稀疏词袋/TF‑IDF 占比较大时效果好），K 由 silhouette/DB 指标与人审结合确定；
    - GMM + BIC/AAIC 选择成分数，适合簇形状接近高斯的情形；
  - 可解释性：对每簇输出“载荷最高”的特征（方向贡献度）与代表性样本工厂地址，生成类型标签。

- 3.4 可视化（生成类似 clustered_semantic_space.pdf）
  - 在正交/白化后的空间上使用 UMAP 或 t‑SNE 降到 2D；
  - 按簇着色，标注簇中心的关键技术模式（如 Clone/CREATE2/Proxy/Token/NFT/DEX/Wallet 等）；
  - 输出到 `experiments/RQ3/clustered_semantic_space.pdf`（或代码里对应的 `factory_clustering_result.pdf`，建议统一为前者）。

---

## 4) 如何让聚类种类尽量保持“正交”

- 以“特征块”为单位进行设计，尽量让不同块反映不同维度：
  - Mechanism（部署机制：CREATE/CREATE2/Clone/Proxy）
  - Access/Control（权限与结构复杂度）
  - Product（产物语义与接口/选择子）
- 流程性手段：
  - 分块标准化 → 分块 PCA → Varimax 旋转（使载荷集中在少数轴上）→ 融合后整体 whitening（协方差≈I）；
  - 也可用 ICA 提取近独立成分，再聚类；
  - 在损失或选择策略上对“跨块相关性”加惩罚（例如在选 K 时优先选择跨块相关性更低的解）；
  - 评估指标：
    - 计算簇心与各特征块子空间投影的互信息/相关性，控制单一块“支配”聚类；
    - 计算簇标签与每个特征块主成分的相关矩阵，目标是对角占优、非对角项尽量小。

---

## 实施计划（可直接落到 `factory_clustering_analysis.py`）

1. 读取地址
   - BigQuery 读取 `rq2_factory_creations`，得出两链工厂地址及基础统计；将地址清单保存到本地 CSV 与 BQ 临时表。
2. 验证与源码
   - 仅使用 Sourcify 批量校验并拉取；未命中直接跳过；本地缓存与（可选）BQ 持久化。
3. 特征构建
   - 工厂源码/字节码机制、权限/结构、业务词袋；
   - 结合其“产物”的运行时代码与选择子分布，聚合为工厂级特征。
4. 正交化与聚类
   - 分块 PCA+Varimax 或 ICA，整体 Whitening；
   - 首选 HDBSCAN，记录噪声点；辅以 KMeans（用 silhouette/DB/BIC 选 K）。
5. 可视化与标注
   - 按链分别运行（ethereum 与 polygon 分开），在“不去重”样本集上进行 UMAP/t‑SNE 到 2D：
     - 生成 `experiments/RQ3/clustered_semantic_space_ethereum.pdf`
     - 生成 `experiments/RQ3/clustered_semantic_space_polygon.pdf`
   - 每簇输出 Top 特征、示例地址，自动命名簇标签并人工微调；图上用点透明度/大小表达频次或创建量，突出占比/影响力。
6. 结果落地
   - 按链分别将 `cluster_id`、`cluster_label`、`top_features`、`sample_addresses` 写入 BQ：`tosem_factory_analysis.factory_clusters_{chain}` 或在同一表中增加分区字段 `chain`，并输出本地 JSON。
   - 提供不去重口径的占比/影响力指标：
     - 按簇的地址占比（样本数占比）。
     - 按簇的创建活动占比：累计 `unique_creations` 与 `total_creations` 的加权占比。
     - 可选：按时间维度（block_date）计算渗透率随时间的变化曲线（每簇）。

---

## 关键启发式与检测细节（摘录）

- EIP‑1167 克隆：运行时代码包含 `0x363d3d373d3d3d363d73...5af43d82803e903d91602b57fd5bf3` 或常见变体。
- CREATE2：源码含 `create2(` 或函数签名出现 `bytes32 salt`；字节码出现 `OPCODE CREATE2 (0xf5)`。
- Proxy/Upgradable：`DELEGATECALL` 的使用、EIP‑1967 常量、Beacon/UUPS 关键词、初始化数据透传等。
- 业务类别（从产物推断）：
  - DEX/Pool（`createPair`/`createPool` 等）
  - Token/NFT（ERC20/721/1155 常见选择子：`0x06fdde03 name()`、`0x95d89b41 symbol()` 等）
  - Wallet/AA（`execTransaction`、`isValidSignature` 等）
  - DAO/Governance（`propose`、`castVote`、`quorum` 等）

---

## 风险与兜底

- 网络/速率限制：批量 + 重试 + 本地缓存；必要时改为离线字节码特征路线。
- 未验证占比较高：加强“产物侧”特征与 EVM 级启发式检测，仍可形成清晰簇。
- 簇解释难：采用 Varimax/ICA 后，输出载荷最大的特征、Top 函数/选择子，辅以人工命名。
- 复合型工厂：允许多标签（主标签 + 次标签），或基于 HDBSCAN 的软分配/成员度解释。

---

## 下一步落地建议

- 在 `factory_clustering_analysis.py` 中：
  - 增加 BigQuery 读取与 Sourcify 抓取模块（带缓存与速率限制）；不再引入 Etherscan/Polygonscan 回落。
  - 将特征分块与正交化流程编码化（PCA+Varimax/ICA + Whitening）。
  - 按链分别执行聚类与可视化，分别输出到：
    - `experiments/RQ3/clustered_semantic_space_ethereum.pdf`
    - `experiments/RQ3/clustered_semantic_space_polygon.pdf`
  - 在不去重样本上训练与评估；如需效率优化，可采用“等价点聚合 + sample_weight=频次”，但报告统计仍按未去重口径输出。
  - 对聚类结果生成 `cluster_type`、`primary_technical`、`primary_business`、`special_patterns`、以及占比/影响力指标。
