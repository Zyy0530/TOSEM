#!/usr/bin/env python3
"""
区块链合约统计脚本
提取五条区块链中的合约总数和unique bytecode的合约总数
支持两种查询类型：
1. 直接查询 (ethereum, polygon) - 使用contracts表
2. JOIN查询 (arbitrum, optimism, avalanche) - 使用transactions和receipts表
"""

import json
import os
from google.cloud import bigquery
from google.oauth2 import service_account
from typing import Dict, List, Tuple
import time
from datetime import datetime

class BlockchainStatistics:
    def __init__(self, config_file: str = "blockchain_config.json"):
        """初始化BigQuery客户端和配置"""
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 从环境变量获取项目ID
        from dotenv import load_dotenv
        load_dotenv()
        
        self.project_id = os.getenv('GOOGLE_CLOUD_PROJECT_ID')
        if not self.project_id:
            raise ValueError("请在.env文件中设置GOOGLE_CLOUD_PROJECT_ID环境变量")
        
        # 设置认证
        credentials_path = "ziyue-wang-26825217908a.json"
        if not os.path.exists(credentials_path):
            raise FileNotFoundError(f"找不到服务账户密钥文件: {credentials_path}")
        
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        self.client = bigquery.Client(project=self.project_id, credentials=credentials)
        
        print(f"✅ BigQuery客户端初始化成功 - 项目: {self.project_id}")
    
    def get_direct_table_stats(self, chain_config: Dict) -> Tuple[int, int]:
        """
        查询直接contracts表的统计信息 (ethereum, polygon)
        返回: (总合约数, unique bytecode数)
        """
        dataset_name = chain_config['dataset_name']
        table_name = chain_config['contracts_table']
        chain_name = chain_config['chain_name']
        
        query = f"""
        SELECT 
            COUNT(*) as total_contracts,
            COUNT(DISTINCT bytecode) as unique_bytecodes
        FROM `{dataset_name}.{table_name}`
        WHERE bytecode IS NOT NULL 
            AND bytecode != '0x'
            AND LENGTH(bytecode) > 2
            AND block_timestamp < '2025-06-01'
        """
        
        print(f"🔍 查询 {chain_name.upper()} 统计信息...")
        print(f"   数据集: {dataset_name}")
        print(f"   查询类型: 直接表查询")
        
        try:
            query_job = self.client.query(query)
            results = query_job.result()
            
            for row in results:
                total_contracts = row.total_contracts
                unique_bytecodes = row.unique_bytecodes
                
                print(f"✅ {chain_name.upper()} 查询完成:")
                print(f"   总合约数: {total_contracts:,}")
                print(f"   Unique Bytecode数: {unique_bytecodes:,}")
                
                return total_contracts, unique_bytecodes
                
        except Exception as e:
            print(f"❌ {chain_name.upper()} 查询失败: {e}")
            return 0, 0
    
    def get_join_table_stats(self, chain_config: Dict) -> Tuple[int, int]:
        """
        查询需要JOIN的表统计信息 (arbitrum, optimism, avalanche)
        返回: (总合约数, unique bytecode数)
        """
        dataset_name = chain_config['dataset_name']
        transactions_table = chain_config['transactions_table']
        receipts_table = chain_config['receipts_table']
        chain_name = chain_config['chain_name']
        
        query = f"""
        WITH contract_creations AS (
            SELECT DISTINCT
                receipts.contract_address,
                transactions.input as bytecode,
                transactions.block_timestamp
            FROM `{dataset_name}.{transactions_table}` AS transactions
            JOIN `{dataset_name}.{receipts_table}` AS receipts
                ON transactions.transaction_hash = receipts.transaction_hash
            WHERE receipts.contract_address IS NOT NULL
                AND transactions.input IS NOT NULL
                AND transactions.input != '0x'
                AND LENGTH(transactions.input) > 2
                AND receipts.status = 1
                AND transactions.block_timestamp < '2025-06-01'
        )
        SELECT 
            COUNT(*) as total_contracts,
            COUNT(DISTINCT bytecode) as unique_bytecodes
        FROM contract_creations
        """
        
        print(f"🔍 查询 {chain_name.upper()} 统计信息...")
        print(f"   数据集: {dataset_name}")
        print(f"   查询类型: JOIN查询 (transactions + receipts)")
        
        try:
            query_job = self.client.query(query)
            results = query_job.result()
            
            for row in results:
                total_contracts = row.total_contracts
                unique_bytecodes = row.unique_bytecodes
                
                print(f"✅ {chain_name.upper()} 查询完成:")
                print(f"   总合约数: {total_contracts:,}")
                print(f"   Unique Bytecode数: {unique_bytecodes:,}")
                
                return total_contracts, unique_bytecodes
                
        except Exception as e:
            print(f"❌ {chain_name.upper()} 查询失败: {e}")
            return 0, 0
    
    def collect_all_statistics(self) -> Dict[str, Dict[str, int]]:
        """收集所有区块链的统计信息"""
        all_stats = {}
        
        print("🚀 开始收集所有区块链统计信息...")
        print("=" * 60)
        
        for chain_config in self.config['blockchain_configs']:
            if chain_config['status'] != 'active':
                continue
                
            chain_name = chain_config['chain_name']
            query_type = chain_config['query_type']
            
            start_time = time.time()
            
            if query_type == 'direct':
                total_contracts, unique_bytecodes = self.get_direct_table_stats(chain_config)
            elif query_type == 'join':
                total_contracts, unique_bytecodes = self.get_join_table_stats(chain_config)
            else:
                print(f"❌ 未知查询类型: {query_type}")
                continue
            
            query_time = time.time() - start_time
            
            all_stats[chain_name] = {
                'total_contracts': total_contracts,
                'unique_bytecodes': unique_bytecodes,
                'query_time_seconds': round(query_time, 2),
                'query_type': query_type,
                'dataset': chain_config['dataset_name']
            }
            
            print(f"⏱️  查询耗时: {query_time:.2f}秒")
            print("-" * 40)
        
        return all_stats
    
    def print_summary(self, stats: Dict[str, Dict[str, int]]):
        """打印统计摘要"""
        print("\n" + "=" * 80)
        print("📊 区块链合约统计摘要")
        print("=" * 80)
        
        total_all_contracts = 0
        total_all_unique = 0
        
        # 打印表头
        print(f"{'区块链':<12} {'总合约数':<15} {'Unique Bytecode':<15} {'查询类型':<10} {'耗时(秒)':<8}")
        print("-" * 70)
        
        # 打印每条链的统计
        for chain_name, data in stats.items():
            total_contracts = data['total_contracts']
            unique_bytecodes = data['unique_bytecodes']
            query_type = data['query_type']
            query_time = data['query_time_seconds']
            
            total_all_contracts += total_contracts
            total_all_unique += unique_bytecodes
            
            print(f"{chain_name.upper():<12} {total_contracts:<15,} {unique_bytecodes:<15,} {query_type:<10} {query_time:<8}")
        
        print("-" * 70)
        print(f"{'总计':<12} {total_all_contracts:<15,} {total_all_unique:<15,}")
        
        # 打印详细分析
        print(f"\n🔍 详细分析:")
        print(f"   • 总合约数量: {total_all_contracts:,}")
        print(f"   • 总Unique Bytecode数: {total_all_unique:,}")
        print(f"   • 平均重复率: {(1 - total_all_unique/total_all_contracts)*100:.2f}%")
        
        # 按查询类型分组
        direct_chains = [name for name, data in stats.items() if data['query_type'] == 'direct']
        join_chains = [name for name, data in stats.items() if data['query_type'] == 'join']
        
        print(f"\n📋 查询方法分布:")
        print(f"   • 直接查询 (contracts表): {', '.join(direct_chains)}")
        print(f"   • JOIN查询 (transactions+receipts): {', '.join(join_chains)}")
    
    def export_results(self, stats: Dict[str, Dict[str, int]], filename: str = None):
        """导出结果到JSON文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"blockchain_statistics_{timestamp}.json"
        
        export_data = {
            'metadata': {
                'collection_date': datetime.now().isoformat(),
                'project_id': self.project_id,
                'total_chains': len(stats),
                'cutoff_date': '2025-06-01'
            },
            'statistics': stats,
            'summary': {
                'total_contracts_all_chains': sum(data['total_contracts'] for data in stats.values()),
                'total_unique_bytecodes_all_chains': sum(data['unique_bytecodes'] for data in stats.values())
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 统计结果已导出到: {filename}")

def main():
    """主函数"""
    print("🏗️ 区块链合约统计工具")
    print("=" * 50)
    
    try:
        # 创建统计收集器
        stats_collector = BlockchainStatistics()
        
        # 收集所有统计信息
        all_statistics = stats_collector.collect_all_statistics()
        
        # 打印摘要
        stats_collector.print_summary(all_statistics)
        
        # 导出结果
        stats_collector.export_results(all_statistics)
        
        print("\n🎉 统计收集完成!")
        
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())