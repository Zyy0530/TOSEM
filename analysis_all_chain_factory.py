#!/usr/bin/env python3
"""
全量扫描区块链合约的工厂合约检测系统（Google BigQuery版）
支持并发处理5个EVM区块链网络的合约字节码分析
数据源：Google BigQuery 公开数据集
存储：Google BigQuery 自定义表
仅处理2025年6月1日之前部署的合约
"""

import os
import sys
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import concurrent.futures
import threading

# Google BigQuery imports
try:
    from google.cloud import bigquery
    from google.oauth2 import service_account
    from google.cloud.exceptions import NotFound
except ImportError:
    print("Error: google-cloud-bigquery not installed. Please run: pip install google-cloud-bigquery")
    sys.exit(1)

from factory_detector import ImprovedFactoryDetector

# =============================================================================
# 配置区域 - 请填写您的Google BigQuery配置
# =============================================================================

# Google BigQuery配置 - 用户需要填写
BIGQUERY_CONFIG = {
    # 项目ID（必填）
    "project_id": "",  # 您的Google Cloud项目ID
    
    # 认证方式选择其一：
    # 方式1：服务账户JSON密钥文件路径（推荐用于生产环境）
    "service_account_key_file": "",  # 服务账户JSON密钥文件的完整路径
    
    # 方式2：使用环境变量GOOGLE_APPLICATION_CREDENTIALS（推荐用于开发环境）
    # 设置环境变量：export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
    
    # 数据集和表配置
    "dataset_id": "tosem_factory_analysis",  # 用于存储分析结果的数据集名称
    "results_table": "factory_analysis_results",  # 存储分析结果的表名
    "progress_table": "analysis_progress",  # 存储进度信息的表名
    
    # 查询配置
    "location": "US",  # BigQuery数据集位置，建议使用US（BigQuery公开数据集位于US）
    "batch_size_months": 1,  # 每次查询的月份数量（1-12）
    "max_parallel_queries": 3,  # 最大并行查询数量（避免配额限制）
    
    # 性能配置
    "use_cache": True,  # 是否使用BigQuery查询缓存
    "dry_run": False,  # 是否启用干运行模式（仅验证查询不执行）
}

# 区块链配置 - 基于Google BigQuery公开数据集
BLOCKCHAIN_CONFIGS = [
    {
        "chain_name": "ethereum",
        "dataset_name": "bigquery-public-data.crypto_ethereum",
        "query_type": "direct",  # 直接从contracts表查询
        "contracts_table": "contracts",
        "genesis_date": "2015-07-30",  # 以太坊主网启动日期
        "status": "active"
    },
    {
        "chain_name": "polygon",
        "dataset_name": "bigquery-public-data.crypto_polygon",
        "query_type": "direct",  # 直接从contracts表查询
        "contracts_table": "contracts",
        "genesis_date": "2020-05-30",  # Polygon主网启动日期
        "status": "active"
    },
    {
        "chain_name": "arbitrum",
        "dataset_name": "bigquery-public-data.goog_blockchain_arbitrum_mainnet_us",
        "query_type": "join",  # 需要JOIN transactions和receipts表
        "transactions_table": "transactions",
        "receipts_table": "receipts",
        "genesis_date": "2021-08-31",  # Arbitrum One主网启动日期
        "status": "active"
    },
    {
        "chain_name": "optimism", 
        "dataset_name": "bigquery-public-data.goog_blockchain_optimism_mainnet_us",
        "query_type": "join",  # 需要JOIN transactions和receipts表
        "transactions_table": "transactions", 
        "receipts_table": "receipts",
        "genesis_date": "2021-12-16",  # Optimism主网启动日期
        "status": "active"
    },
    {
        "chain_name": "avalanche",
        "dataset_name": "bigquery-public-data.goog_blockchain_avalanche_mainnet_us", 
        "query_type": "join",  # 需要JOIN transactions和receipts表
        "transactions_table": "transactions",
        "receipts_table": "receipts", 
        "genesis_date": "2020-09-23",  # Avalanche C-Chain启动日期
        "status": "active"
    }
]

# 分析配置
ANALYSIS_CONFIG = {
    "cutoff_date": "2025-06-01",  # 分析截止日期
    "max_workers": 5,  # 最大工作线程数
    "batch_save_size": 1000,  # 批量保存大小
    "retry_attempts": 3,  # 查询失败重试次数
    "retry_delay": 60,  # 重试延迟（秒）
}

# =============================================================================

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bigquery_factory_analysis.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ContractData:
    """合约数据结构"""
    chain: str
    address: str
    bytecode: str
    created_at: datetime
    block_number: Optional[int] = None
    tx_hash: Optional[str] = None


@dataclass 
class AnalysisResult:
    """分析结果数据结构"""
    chain: str
    address: str
    is_factory: bool
    is_create: bool
    is_create2: bool  
    is_both: bool
    analysis_success: bool
    analysis_time: int  # 毫秒
    processed_at: datetime


class BigQueryManager:
    """Google BigQuery管理器"""
    
    def __init__(self, config: Dict):
        """
        初始化BigQuery客户端
        
        Args:
            config: BigQuery配置字典
        """
        self.config = config
        self.project_id = config['project_id']
        self.dataset_id = config['dataset_id'] 
        self.location = config['location']
        
        if not self.project_id:
            raise ValueError("项目ID不能为空，请在BIGQUERY_CONFIG中设置project_id")
        
        # 初始化BigQuery客户端
        self.client = self._create_client()
        
        # 初始化数据集和表
        self._setup_dataset()
        
    def _create_client(self) -> bigquery.Client:
        """创建BigQuery客户端"""
        try:
            # 方式1：使用服务账户密钥文件
            key_file = self.config.get('service_account_key_file')
            if key_file and os.path.exists(key_file):
                credentials = service_account.Credentials.from_service_account_file(key_file)
                client = bigquery.Client(
                    project=self.project_id,
                    credentials=credentials,
                    location=self.location
                )
                logger.info(f"Using service account key file: {key_file}")
                
            # 方式2：使用环境变量GOOGLE_APPLICATION_CREDENTIALS
            elif os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
                client = bigquery.Client(project=self.project_id, location=self.location)
                logger.info("Using GOOGLE_APPLICATION_CREDENTIALS environment variable")
                
            else:
                # 方式3：使用默认凭据（如果运行在Google Cloud上）
                client = bigquery.Client(project=self.project_id, location=self.location)
                logger.info("Using default credentials")
            
            # 测试连接
            client.get_dataset(client.dataset('bigquery-public-data'))
            logger.info(f"BigQuery client initialized successfully for project: {self.project_id}")
            return client
            
        except Exception as e:
            logger.error(f"Failed to initialize BigQuery client: {e}")
            raise
    
    def _setup_dataset(self):
        """设置数据集和表"""
        try:
            # 创建数据集（如果不存在）
            dataset_ref = self.client.dataset(self.dataset_id)
            try:
                self.client.get_dataset(dataset_ref)
                logger.info(f"Dataset {self.dataset_id} already exists")
            except NotFound:
                dataset = bigquery.Dataset(dataset_ref)
                dataset.location = self.location
                dataset = self.client.create_dataset(dataset)
                logger.info(f"Created dataset {self.dataset_id}")
            
            # 创建结果表
            self._create_results_table()
            
            # 创建进度表
            self._create_progress_table()
            
        except Exception as e:
            logger.error(f"Failed to setup dataset: {e}")
            raise
    
    def _create_results_table(self):
        """创建分析结果表"""
        table_id = f"{self.project_id}.{self.dataset_id}.{self.config['results_table']}"
        
        schema = [
            bigquery.SchemaField("chain", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("address", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("is_factory", "BOOLEAN", mode="REQUIRED"),
            bigquery.SchemaField("is_create", "BOOLEAN", mode="NULLABLE"), 
            bigquery.SchemaField("is_create2", "BOOLEAN", mode="NULLABLE"),
            bigquery.SchemaField("is_both", "BOOLEAN", mode="NULLABLE"),
            bigquery.SchemaField("analysis_success", "BOOLEAN", mode="REQUIRED"),
            bigquery.SchemaField("analysis_time", "INTEGER", mode="REQUIRED"),  # 毫秒
            bigquery.SchemaField("processed_at", "TIMESTAMP", mode="REQUIRED"),
        ]
        
        try:
            table = bigquery.Table(table_id, schema=schema)
            table.clustering_fields = ["chain", "address"] 
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="processed_at"
            )
            table = self.client.create_table(table, exists_ok=True)
            logger.info(f"Results table created/verified: {table_id}")
            
        except Exception as e:
            logger.error(f"Failed to create results table: {e}")
            raise
    
    def _create_progress_table(self):
        """创建进度表"""
        table_id = f"{self.project_id}.{self.dataset_id}.{self.config['progress_table']}"
        
        schema = [
            bigquery.SchemaField("chain", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("start_date", "DATE", mode="REQUIRED"),
            bigquery.SchemaField("end_date", "DATE", mode="REQUIRED"),
            bigquery.SchemaField("status", "STRING", mode="REQUIRED"),  # 'completed', 'in_progress', 'failed'
            bigquery.SchemaField("contracts_processed", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("factories_found", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("processing_time_ms", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
        ]
        
        try:
            table = bigquery.Table(table_id, schema=schema)
            table.clustering_fields = ["chain", "start_date"]
            table = self.client.create_table(table, exists_ok=True)
            logger.info(f"Progress table created/verified: {table_id}")
            
        except Exception as e:
            logger.error(f"Failed to create progress table: {e}")
            raise
    
    def get_processed_periods(self, chain: str) -> List[Tuple[datetime, datetime]]:
        """获取已处理的时间段"""
        query = f"""
        SELECT start_date, end_date
        FROM `{self.project_id}.{self.dataset_id}.{self.config['progress_table']}`
        WHERE chain = @chain AND status = 'completed'
        ORDER BY start_date DESC
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("chain", "STRING", chain)
            ],
            use_query_cache=self.config['use_cache']
        )
        
        try:
            results = self.client.query(query, job_config=job_config)
            processed_periods = []
            for row in results:
                start_date = datetime.combine(row.start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
                end_date = datetime.combine(row.end_date, datetime.min.time()).replace(tzinfo=timezone.utc)
                processed_periods.append((start_date, end_date))
            
            logger.info(f"Found {len(processed_periods)} completed periods for {chain}")
            return processed_periods
            
        except Exception as e:
            logger.error(f"Failed to get processed periods for {chain}: {e}")
            return []
    
    def mark_period_in_progress(self, chain: str, start_date: datetime, end_date: datetime):
        """标记时间段为处理中"""
        query = f"""
        INSERT INTO `{self.project_id}.{self.dataset_id}.{self.config['progress_table']}`
        (chain, start_date, end_date, status, updated_at)
        VALUES (@chain, @start_date, @end_date, 'in_progress', CURRENT_TIMESTAMP())
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("chain", "STRING", chain),
                bigquery.ScalarQueryParameter("start_date", "DATE", start_date.date()),
                bigquery.ScalarQueryParameter("end_date", "DATE", end_date.date())
            ]
        )
        
        try:
            self.client.query(query, job_config=job_config)
            logger.debug(f"Marked period {start_date.date()} to {end_date.date()} as in_progress for {chain}")
            
        except Exception as e:
            logger.error(f"Failed to mark period in_progress for {chain}: {e}")
    
    def update_period_completed(self, chain: str, start_date: datetime, end_date: datetime,
                               contracts_processed: int, factories_found: int, processing_time_ms: int):
        """更新时间段为已完成"""
        query = f"""
        UPDATE `{self.project_id}.{self.dataset_id}.{self.config['progress_table']}`
        SET 
            status = 'completed',
            contracts_processed = @contracts_processed,
            factories_found = @factories_found, 
            processing_time_ms = @processing_time_ms,
            updated_at = CURRENT_TIMESTAMP()
        WHERE chain = @chain 
            AND start_date = @start_date 
            AND end_date = @end_date
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("chain", "STRING", chain),
                bigquery.ScalarQueryParameter("start_date", "DATE", start_date.date()),
                bigquery.ScalarQueryParameter("end_date", "DATE", end_date.date()),
                bigquery.ScalarQueryParameter("contracts_processed", "INTEGER", contracts_processed),
                bigquery.ScalarQueryParameter("factories_found", "INTEGER", factories_found),
                bigquery.ScalarQueryParameter("processing_time_ms", "INTEGER", processing_time_ms)
            ]
        )
        
        try:
            self.client.query(query, job_config=job_config)
            logger.debug(f"Marked period {start_date.date()} to {end_date.date()} as completed for {chain}")
            
        except Exception as e:
            logger.error(f"Failed to mark period completed for {chain}: {e}")
    
    def save_analysis_results(self, results: List[AnalysisResult]) -> int:
        """批量保存分析结果到BigQuery"""
        if not results:
            return 0
        
        table_id = f"{self.project_id}.{self.dataset_id}.{self.config['results_table']}"
        
        # 转换数据
        rows_to_insert = []
        for result in results:
            rows_to_insert.append({
                "chain": result.chain,
                "address": result.address,
                "is_factory": result.is_factory,
                "is_create": result.is_create if result.is_factory else None,
                "is_create2": result.is_create2 if result.is_factory else None,
                "is_both": result.is_both if result.is_factory else None,
                "analysis_success": result.analysis_success,
                "analysis_time": result.analysis_time,
                "processed_at": result.processed_at
            })
        
        try:
            # 执行插入操作
            errors = self.client.insert_rows_json(
                self.client.get_table(table_id), 
                rows_to_insert,
                row_ids=[f"{row['chain']}_{row['address']}" for row in rows_to_insert]  # 用于去重
            )
            
            if errors:
                logger.error(f"Failed to insert some rows: {errors}")
                return len(results) - len(errors)
            else:
                logger.info(f"Successfully saved {len(results)} analysis results")
                return len(results)
                
        except Exception as e:
            logger.error(f"Failed to save analysis results: {e}")
            return 0


class BigQueryContractFetcher:
    """Google BigQuery合约数据获取器"""
    
    def __init__(self, bigquery_manager: BigQueryManager):
        """初始化合约获取器"""
        self.bq = bigquery_manager
        self.client = bigquery_manager.client
        
    def _create_ethereum_query(self, start_date: datetime, end_date: datetime) -> str:
        """创建以太坊合约查询SQL"""
        return f"""
        SELECT 
            address,
            bytecode as code,
            block_timestamp as created_at,
            block_number,
            transaction_hash as tx_hash
        FROM `bigquery-public-data.crypto_ethereum.contracts`
        WHERE block_timestamp >= @start_date
            AND block_timestamp < @end_date
            AND bytecode IS NOT NULL
            AND LENGTH(bytecode) > 0
        ORDER BY block_timestamp DESC
        """
    
    def _create_polygon_query(self, start_date: datetime, end_date: datetime) -> str:
        """创建Polygon合约查询SQL"""  
        return f"""
        SELECT 
            address,
            bytecode as code,
            block_timestamp as created_at,
            block_number,
            transaction_hash as tx_hash
        FROM `bigquery-public-data.crypto_polygon.contracts`
        WHERE block_timestamp >= @start_date
            AND block_timestamp < @end_date
            AND bytecode IS NOT NULL
            AND LENGTH(bytecode) > 0
        ORDER BY block_timestamp DESC
        """
        
    def _create_join_query(self, dataset: str, start_date: datetime, end_date: datetime) -> str:
        """创建需要JOIN的区块链查询SQL（Arbitrum, Optimism, Avalanche）"""
        return f"""
        SELECT 
            r.contract_address as address,
            t.input as code,  -- 合约创建交易的input包含字节码
            t.block_timestamp as created_at,
            t.block_number,
            t.hash as tx_hash
        FROM `{dataset}.transactions` t
        JOIN `{dataset}.receipts` r ON t.hash = r.transaction_hash
        WHERE t.to_address IS NULL  -- 合约创建交易
            AND r.contract_address IS NOT NULL
            AND r.status = 1  -- 成功的交易
            AND t.input IS NOT NULL
            AND LENGTH(t.input) > 2  -- 确保有实际的字节码内容（不只是'0x'）
            AND t.block_timestamp >= @start_date
            AND t.block_timestamp < @end_date
        ORDER BY t.block_timestamp DESC
        """
    
    def fetch_contracts(self, chain_config: Dict, start_date: datetime, end_date: datetime) -> List[ContractData]:
        """
        获取指定时间段内的合约数据
        
        Args:
            chain_config: 区块链配置
            start_date: 开始时间
            end_date: 结束时间
            
        Returns:
            合约数据列表
        """
        chain_name = chain_config['chain_name']
        query_type = chain_config['query_type']
        
        # 根据不同的查询类型生成SQL
        if chain_name == 'ethereum':
            query = self._create_ethereum_query(start_date, end_date)
        elif chain_name == 'polygon':
            query = self._create_polygon_query(start_date, end_date)
        elif query_type == 'join':
            query = self._create_join_query(chain_config['dataset_name'], start_date, end_date)
        else:
            raise ValueError(f"Unsupported query type for {chain_name}: {query_type}")
        
        # 设置查询参数
        job_config = bigquery.QueryJobConfig(
            use_query_cache=self.bq.config['use_cache'],
            dry_run=self.bq.config['dry_run']
        )
        
        # 所有查询都需要时间参数（现在Polygon也需要了）
        job_config.query_parameters = [
            bigquery.ScalarQueryParameter("start_date", "TIMESTAMP", start_date),
            bigquery.ScalarQueryParameter("end_date", "TIMESTAMP", end_date)
        ]
        
        try:
            logger.info(f"Fetching {chain_name} contracts from {start_date.date()} to {end_date.date()}")
            
            if self.bq.config['dry_run']:
                logger.info(f"DRY RUN - Query for {chain_name}: {query}")
                return []
            
            # 执行查询
            query_job = self.client.query(query, job_config=job_config)
            results = query_job.result()
            
            # 转换结果
            contracts = []
            for row in results:
                try:
                    # 处理创建时间
                    created_at = row.get('created_at')
                    if created_at is None:
                        created_at = datetime.now(timezone.utc)
                    elif isinstance(created_at, str):
                        created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    
                    contract = ContractData(
                        chain=chain_name,
                        address=row['address'].lower(),  # 统一转为小写
                        bytecode=row['code'],
                        created_at=created_at,
                        block_number=row.get('block_number'),
                        tx_hash=row.get('tx_hash')
                    )
                    contracts.append(contract)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse contract data: {e}")
                    continue
            
            logger.info(f"Fetched {len(contracts)} contracts from {chain_name}")
            return contracts
            
        except Exception as e:
            logger.error(f"Failed to fetch contracts from {chain_name}: {e}")
            return []


class FactoryAnalysisSystem:
    """基于Google BigQuery的工厂合约分析系统"""
    
    def __init__(self):
        """初始化分析系统"""
        # 验证配置
        if not BIGQUERY_CONFIG['project_id']:
            raise ValueError("请在BIGQUERY_CONFIG中设置project_id")
        
        # 初始化组件
        self.bq = BigQueryManager(BIGQUERY_CONFIG)
        self.fetcher = BigQueryContractFetcher(self.bq)
        self.detector = ImprovedFactoryDetector()
        
        # 统计信息
        self.stats = {
            'total_processed': 0,
            'total_factories': 0,
            'create_only': 0,
            'create2_only': 0,
            'both_create_create2': 0,
            'errors': 0,
            'start_time': None
        }
        
        # 线程安全锁
        self._lock = threading.Lock()
    
    def _update_stats(self, processed: int = 0, factories: int = 0, 
                     create_only: int = 0, create2_only: int = 0, 
                     both: int = 0, errors: int = 0):
        """线程安全地更新统计信息"""
        with self._lock:
            self.stats['total_processed'] += processed
            self.stats['total_factories'] += factories
            self.stats['create_only'] += create_only
            self.stats['create2_only'] += create2_only
            self.stats['both_create_create2'] += both
            self.stats['errors'] += errors
    
    def analyze_contract(self, contract: ContractData) -> AnalysisResult:
        """
        分析单个合约
        
        Args:
            contract: 合约数据
            
        Returns:
            分析结果
        """
        start_time = time.perf_counter()
        
        try:
            # 使用检测器分析
            detection_result = self.detector.detect_factory_contract(contract.bytecode)
            
            analysis_time = int((time.perf_counter() - start_time) * 1000)  # 毫秒
            
            # 解析结果
            is_factory = detection_result.is_factory_contract
            factory_type = detection_result.factory_type
            
            result = AnalysisResult(
                chain=contract.chain,
                address=contract.address,
                is_factory=is_factory,
                is_create=factory_type in ['CREATE_ONLY', 'BOTH_CREATE_CREATE2'] if is_factory else False,
                is_create2=factory_type in ['CREATE2_ONLY', 'BOTH_CREATE_CREATE2'] if is_factory else False,
                is_both=factory_type == 'BOTH_CREATE_CREATE2' if is_factory else False,
                analysis_success=True,
                analysis_time=analysis_time,
                processed_at=datetime.now(timezone.utc)
            )
            
            return result
            
        except Exception as e:
            analysis_time = int((time.perf_counter() - start_time) * 1000)
            logger.error(f"Failed to analyze contract {contract.address} on {contract.chain}: {e}")
            
            return AnalysisResult(
                chain=contract.chain,
                address=contract.address,
                is_factory=False,
                is_create=False,
                is_create2=False,
                is_both=False,
                analysis_success=False,
                analysis_time=analysis_time,
                processed_at=datetime.now(timezone.utc)
            )
    
    def generate_time_periods(self, chain_config: Dict) -> List[Tuple[datetime, datetime]]:
        """
        生成需要处理的时间段
        
        Args:
            chain_config: 区块链配置
            
        Returns:
            时间段列表 (start_date, end_date)
        """
        chain_name = chain_config['chain_name']
        genesis_date = datetime.fromisoformat(chain_config['genesis_date']).replace(tzinfo=timezone.utc)
        cutoff_date = datetime.fromisoformat(ANALYSIS_CONFIG['cutoff_date']).replace(tzinfo=timezone.utc)
        
        # 获取已处理的时间段
        processed_periods = self.bq.get_processed_periods(chain_name)
        processed_set = set(processed_periods)
        
        # 生成所有需要处理的月份
        all_periods = []
        current_date = cutoff_date
        batch_months = BIGQUERY_CONFIG['batch_size_months']
        
        while current_date > genesis_date:
            # 计算批次开始时间
            start_date = current_date - timedelta(days=30 * batch_months)
            if start_date < genesis_date:
                start_date = genesis_date
            
            period = (start_date, current_date)
            
            # 检查是否已处理
            if period not in processed_set:
                all_periods.append(period)
            
            current_date = start_date
        
        logger.info(f"Generated {len(all_periods)} periods to process for {chain_name}")
        return all_periods
    
    def process_chain_period(self, chain_config: Dict, start_date: datetime, end_date: datetime) -> Dict:
        """
        处理单个区块链的单个时间段
        
        Args:
            chain_config: 区块链配置
            start_date: 开始时间
            end_date: 结束时间
            
        Returns:
            处理结果统计
        """
        chain_name = chain_config['chain_name']
        period_start_time = time.perf_counter()
        
        logger.info(f"Processing {chain_name}: {start_date.date()} to {end_date.date()}")
        
        try:
            # 标记为处理中
            self.bq.mark_period_in_progress(chain_name, start_date, end_date)
            
            # 获取合约数据
            contracts = self.fetcher.fetch_contracts(chain_config, start_date, end_date)
            
            if not contracts:
                logger.info(f"No contracts found for {chain_name} in period {start_date.date()} to {end_date.date()}")
                # 标记为完成（即使没有合约）
                processing_time = int((time.perf_counter() - period_start_time) * 1000)
                self.bq.update_period_completed(chain_name, start_date, end_date, 0, 0, processing_time)
                return {'processed': 0, 'factories': 0, 'errors': 0}
            
            # 分析合约
            results = []
            factories_found = 0
            errors = 0
            
            for i, contract in enumerate(contracts):
                try:
                    result = self.analyze_contract(contract)
                    results.append(result)
                    
                    if result.is_factory:
                        factories_found += 1
                    
                    if not result.analysis_success:
                        errors += 1
                    
                    # 批量保存
                    if len(results) >= ANALYSIS_CONFIG['batch_save_size']:
                        saved = self.bq.save_analysis_results(results)
                        logger.info(f"{chain_name}: Saved batch of {saved} results")
                        results = []
                    
                    # 进度报告
                    if (i + 1) % 100 == 0:
                        logger.info(f"{chain_name}: Processed {i + 1}/{len(contracts)} contracts")
                        
                except Exception as e:
                    logger.error(f"Error processing contract {contract.address}: {e}")
                    errors += 1
            
            # 保存剩余结果
            if results:
                saved = self.bq.save_analysis_results(results)
                logger.info(f"{chain_name}: Saved final batch of {saved} results")
            
            # 更新统计
            self._update_stats(
                processed=len(contracts),
                factories=factories_found,
                errors=errors
            )
            
            # 标记时间段为完成
            processing_time = int((time.perf_counter() - period_start_time) * 1000)
            self.bq.update_period_completed(chain_name, start_date, end_date, 
                                          len(contracts), factories_found, processing_time)
            
            logger.info(f"✓ Completed {chain_name}: {len(contracts)} contracts, {factories_found} factories")
            
            return {
                'processed': len(contracts),
                'factories': factories_found, 
                'errors': errors
            }
            
        except Exception as e:
            logger.error(f"Failed to process {chain_name} period {start_date.date()} to {end_date.date()}: {e}")
            return {'processed': 0, 'factories': 0, 'errors': 1}
    
    def process_chain(self, chain_config: Dict) -> Dict:
        """
        处理单个区块链的所有时间段
        
        Args:
            chain_config: 区块链配置
            
        Returns:
            处理结果统计
        """
        chain_name = chain_config['chain_name']
        logger.info(f"🚀 Starting processing for {chain_name}")
        
        try:
            # 生成时间段
            time_periods = self.generate_time_periods(chain_config)
            
            if not time_periods:
                logger.info(f"No periods to process for {chain_name}")
                return {'processed': 0, 'factories': 0, 'errors': 0}
            
            # 串行处理各个时间段（避免BigQuery并发限制）
            total_stats = {'processed': 0, 'factories': 0, 'errors': 0}
            
            for start_date, end_date in time_periods:
                period_stats = self.process_chain_period(chain_config, start_date, end_date)
                
                total_stats['processed'] += period_stats['processed']
                total_stats['factories'] += period_stats['factories'] 
                total_stats['errors'] += period_stats['errors']
                
                # 短暂休息避免配额限制
                time.sleep(1)
            
            logger.info(f"🎯 Completed all periods for {chain_name}: "
                       f"{total_stats['processed']} contracts, {total_stats['factories']} factories")
            
            return total_stats
            
        except Exception as e:
            logger.error(f"💥 Failed to process {chain_name}: {e}")
            return {'processed': 0, 'factories': 0, 'errors': 1}
    
    def run_concurrent_analysis(self):
        """运行并发分析"""
        logger.info("🌟 Starting BigQuery Factory Contract Analysis System")
        logger.info(f"Configuration: {len(BLOCKCHAIN_CONFIGS)} chains, cutoff date: {ANALYSIS_CONFIG['cutoff_date']}")
        
        self.stats['start_time'] = time.time()
        
        # 验证配置
        active_chains = [cfg for cfg in BLOCKCHAIN_CONFIGS if cfg.get('status') == 'active']
        logger.info(f"Active blockchain configurations: {len(active_chains)}")
        
        for chain_config in active_chains:
            chain_name = chain_config['chain_name']
            logger.info(f"  → {chain_name}: {chain_config['dataset_name']} ({chain_config['query_type']})")
        
        # 并发处理区块链
        max_workers = min(len(active_chains), ANALYSIS_CONFIG['max_workers'])
        logger.info(f"Starting concurrent processing with {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            future_to_chain = {
                executor.submit(self.process_chain, chain_config): chain_config['chain_name']
                for chain_config in active_chains
            }
            
            # 收集结果
            for future in as_completed(future_to_chain):
                chain_name = future_to_chain[future]
                try:
                    result = future.result()
                    logger.info(f"✅ {chain_name} completed: {result}")
                except Exception as e:
                    logger.error(f"❌ {chain_name} failed: {e}")
        
        # 打印最终统计
        self.print_final_stats()
    
    def print_final_stats(self):
        """打印最终统计"""
        total_time = time.time() - self.stats['start_time']
        
        logger.info("=" * 80)
        logger.info("🏆 BIGQUERY FACTORY CONTRACT ANALYSIS COMPLETED")
        logger.info("=" * 80)
        logger.info(f"📊 Total contracts processed: {self.stats['total_processed']:,}")
        logger.info(f"🏭 Total factory contracts found: {self.stats['total_factories']:,}")
        logger.info(f"📈 CREATE only factories: {self.stats['create_only']:,}")
        logger.info(f"📈 CREATE2 only factories: {self.stats['create2_only']:,}")
        logger.info(f"📈 Both CREATE and CREATE2: {self.stats['both_create_create2']:,}")
        logger.info(f"❌ Analysis errors: {self.stats['errors']:,}")
        logger.info(f"⏱️  Total analysis time: {total_time:.2f} seconds")
        
        if self.stats['total_processed'] > 0:
            avg_time = (total_time / self.stats['total_processed']) * 1000
            factory_rate = (self.stats['total_factories'] / self.stats['total_processed']) * 100
            logger.info(f"⚡ Average time per contract: {avg_time:.3f}ms")
            logger.info(f"📊 Factory contract rate: {factory_rate:.2f}%")
        
        logger.info("=" * 80)


def main():
    """主函数"""
    try:
        # 验证配置
        if not BIGQUERY_CONFIG['project_id']:
            logger.error("❌ 请在代码顶部的BIGQUERY_CONFIG中设置project_id")
            logger.error("💡 同时请确保已配置Google Cloud认证:")
            logger.error("   方式1：设置service_account_key_file路径")
            logger.error("   方式2：设置环境变量GOOGLE_APPLICATION_CREDENTIALS") 
            logger.error("   方式3：在Google Cloud环境中使用默认凭据")
            sys.exit(1)
        
        logger.info("🎯 Google BigQuery Factory Contract Analysis System")
        logger.info(f"📋 Project: {BIGQUERY_CONFIG['project_id']}")
        logger.info(f"📊 Dataset: {BIGQUERY_CONFIG['dataset_id']}")
        logger.info(f"🌍 Location: {BIGQUERY_CONFIG['location']}")
        
        # 创建并运行分析系统
        analyzer = FactoryAnalysisSystem()
        analyzer.run_concurrent_analysis()
        
    except KeyboardInterrupt:
        logger.info("⚠️  Analysis interrupted by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()