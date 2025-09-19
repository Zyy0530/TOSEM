现在，我们需要进行第二个实验，用于统计两条区块链（Eth和Poly）

- 每日部署的工厂合约的部署数量（截止到2025年6月）
- 所有工厂合约每日交易的数量（create交易）（不需要基于跑的代码）
- 字节码重复的数量从低到高，以及对应的CDF图（需要基于实验）工厂合约的字节码重复数量分布
- 每个工厂合约交易数量从低到高，以及对应的CDF图（需要基于实验）工厂合约的交易数量分布

## 任务1
和rq2_factory_detection.py不同的是，我们不再依赖detector对工厂合约进行区分，而是直接从BigQuery表中获取工厂合约的数量
具体来说：
Bigquery的数据集：https://console.cloud.google.com/bigquery?ws=!1m4!1m3!3m2!1sbigquery-public-data!2scrypto_ethereum
以及https://console.cloud.google.com/bigquery?ws=!1m4!1m3!3m2!1sbigquery-public-data!2scrypto_polygon数据集
他们分别包含了traces表，你可以通过这个表的trace_type是否为'create',来判断是否为工厂合约
将结果保存到BigQuery的factory表

## 任务2
为了完成下面的四个SQL查询：

- 每日部署的工厂合约的部署数量（截止到2025年6月）
- 所有工厂合约每日交易的数量（create交易）（不需要基于跑的代码）
- 字节码重复的数量从低到高，以及对应的CDF图（需要基于实验）工厂合约的字节码重复数量分布
- 每个工厂合约交易数量从低到高，以及对应的CDF图工厂合约的交易数量分布


首先，你需要基于这两条链的工厂合约。 再通过sql查询，得到数量 / 数量分布
将结果保存到本地csv文件


## 任务3

最后，有了具体的数据之后，编写脚本，阅读这些CSV文件，分别绘制四幅pdf图，用于描述上述四个数据 / 数据分布


现在，你需要先完成任务1: 识别工厂合约，并创建一个factory表（保存到BigQuery中），记录两条区块链的所有工厂合约。
