现在，我们需要进行第二个实验，用于统计两条区块链（Eth和Poly）
工厂合约的数量，以及相对于总体的变化趋势

- 每日部署的工厂合约的部署数量（截止到2025年6月）[区分create / create2 / both] 堆叠柱状图（需跑一遍实验）
- 所有工厂合约每日交易的数量（create交易）（不需要基于跑的代码）
- 字节码重复的数量从低到高，以及对应的CDF图（需要基于实验）工厂合约的字节码重复数量分布
- 每个工厂合约交易数量从低到高，以及对应的CDF图（需要基于实验）工厂合约的交易数量分布
- 


为了完成上述四部分内容的撰写，我们需要使用[factory_detector.py](../../factory_detector.py)进行实验
这个实验具体来说，需要通过google bigquery进行获取。
```json
{
      "chain_name": "ethereum",
      "display_name": "Ethereum Mainnet", 
      "dataset_name": "bigquery-public-data.crypto_ethereum",
      "query_type": "direct",
      "contracts_table": "contracts",
      "genesis_date": "2015-07-30",
      "status": "active",
      "description": "Ethereum mainnet contracts from crypto_ethereum dataset",
      "query_fields": {
        "address_field": "address",
        "bytecode_field": "bytecode", 
        "timestamp_field": "block_timestamp",
        "block_number_field": "block_number",
        "tx_hash_field": "transaction_hash"
      }
    },
    {
      "chain_name": "polygon",
      "display_name": "Polygon Mainnet",
      "dataset_name": "bigquery-public-data.crypto_polygon", 
      "query_type": "direct",
      "contracts_table": "contracts",
      "genesis_date": "2020-05-30",
      "status": "active", 
      "description": "Polygon mainnet contracts from crypto_polygon dataset",
      "query_fields": {
        "address_field": "address",
        "bytecode_field": "bytecode",
        "timestamp_field": "block_timestamp",
        "block_number_field": "block_number",
        "tx_hash_field": "transaction_hash"
      }
    },
```
- 这是具体的两条区块链的合约表，你需要使用bigquery获取部署（创建）时间在2025年6月1日之前的所有的合约字节码（batch）
- 然后使用factory detector运行这些字节码
- 然后在BigQuery新建一个新表，记录这两条链的检测结果
- 列需要包含下面几个字段：chain，address，is_factory, is_create2_only, is_create_only, is_both


你需要首先在RQ2中编写实验脚本，这个脚本需要有清晰的执行进展记录（执行了 x% 的合约）
以及这个会首先检查当前已经执行了多少，每次运行，只会运行未被记录的合约，而不会运行已执行的合约

你需要认真思考，帮我先编写这个脚本，并做好开始第二个实验的准备
