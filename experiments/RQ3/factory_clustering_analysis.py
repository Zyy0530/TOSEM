#!/usr/bin/env python3
"""
Factory Contract Clustering Analysis
工厂合约聚类分析 - 基于语法、语义、结构特征的多维聚类
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append('../..')
from ground_truth_dataset import GroundTruthDatasetBuilder

class FactoryContractFeatureExtractor:
    """工厂合约特征提取器"""
    
    def __init__(self):
        # 预定义的业务领域关键词
        self.business_keywords = {
            'finance': ['swap', 'pool', 'liquidity', 'lending', 'borrow', 'yield', 'farm', 'stake', 
                       'token', 'coin', 'defi', 'exchange', 'trade', 'price', 'fee', 'reward'],
            'nft': ['nft', 'token', 'metadata', 'uri', 'collection', 'mint', 'burn', 'transfer',
                   'erc721', 'erc1155', 'opensea', 'marketplace', 'royalty', 'game', 'art'],
            'infrastructure': ['proxy', 'implementation', 'beacon', 'registry', 'manager', 'admin',
                             'upgrade', 'delegate', 'forwarding', 'multicall', 'batch', 'utility'],
            'governance': ['dao', 'governance', 'proposal', 'vote', 'quorum', 'timelock', 'treasury',
                          'council', 'member', 'delegate', 'multisig', 'permission', 'role']
        }
        
        # 技术模式关键词
        self.technical_patterns = {
            'clone_factory': ['clone', 'clonefactory', 'minimal_proxy', 'eip1167'],
            'create2_factory': ['create2', 'deterministic', 'salt', 'predict'],
            'proxy_factory': ['proxy', 'implementation', 'beacon', 'upgrade'],
            'registry_factory': ['registry', 'register', 'mapping', 'directory']
        }
        
    def extract_features(self, source_code: str) -> Dict:
        """提取单个合约的多维特征"""
        features = {}
        source_lower = source_code.lower()
        
        # 1. 语法特征
        features.update(self._extract_syntactic_features(source_code, source_lower))
        
        # 2. 语义特征  
        features.update(self._extract_semantic_features(source_code, source_lower))
        
        # 3. 结构特征
        features.update(self._extract_structural_features(source_code, source_lower))
        
        return features
    
    def _extract_syntactic_features(self, source_code: str, source_lower: str) -> Dict:
        """提取语法特征"""
        features = {}
        
        # 函数命名模式
        create_functions = len(re.findall(r'function\s+create\w*', source_lower))
        deploy_functions = len(re.findall(r'function\s+deploy\w*', source_lower))
        new_statements = len(re.findall(r'new\s+\w+', source_lower))
        clone_functions = len(re.findall(r'function\s+clone\w*', source_lower))
        
        features.update({
            'create_functions': create_functions,
            'deploy_functions': deploy_functions, 
            'new_statements': new_statements,
            'clone_functions': clone_functions,
            'total_factory_methods': create_functions + deploy_functions + clone_functions
        })
        
        # 修饰符使用
        modifiers = ['onlyowner', 'payable', 'external', 'public', 'internal', 'private']
        for modifier in modifiers:
            features[f'has_{modifier}'] = 1 if modifier in source_lower else 0
            
        # 事件定义
        events = len(re.findall(r'event\s+\w+', source_lower))
        features['total_events'] = events
        
        return features
    
    def _extract_semantic_features(self, source_code: str, source_lower: str) -> Dict:
        """提取语义特征"""
        features = {}
        
        # 业务领域特征
        for domain, keywords in self.business_keywords.items():
            domain_score = sum(1 for keyword in keywords if keyword in source_lower)
            features[f'{domain}_score'] = domain_score
            
        # 标准库使用
        standard_libs = ['erc20', 'erc721', 'erc1155', 'ownable', 'pausable', 'reentrancy']
        for lib in standard_libs:
            features[f'uses_{lib}'] = 1 if lib in source_lower else 0
            
        # OpenZeppelin 导入
        features['uses_openzeppelin'] = 1 if 'openzeppelin' in source_lower else 0
        
        return features
    
    def _extract_structural_features(self, source_code: str, source_lower: str) -> Dict:
        """提取结构特征"""
        features = {}
        
        # CREATE操作模式
        features['has_create'] = 1 if 'create(' in source_lower else 0
        features['has_create2'] = 1 if 'create2(' in source_lower else 0
        
        # 技术模式
        for pattern, keywords in self.technical_patterns.items():
            pattern_score = sum(1 for keyword in keywords if keyword in source_lower)
            features[f'{pattern}_score'] = pattern_score
            
        # 状态管理复杂度
        mappings = len(re.findall(r'mapping\s*\(', source_lower))
        arrays = len(re.findall(r'\[\]\s+', source_code))
        features['state_complexity'] = mappings + arrays
        
        # 合约大小 (代码行数的代理)
        features['code_length'] = len(source_code)
        features['code_lines'] = source_code.count('\n')
        
        return features

class FactoryClusteringAnalyzer:
    """工厂合约聚类分析器"""
    
    def __init__(self):
        self.feature_extractor = FactoryContractFeatureExtractor()
        self.scaler = StandardScaler()
        
    def analyze_contracts(self, contracts_data: List[Dict]) -> Dict:
        """执行完整的聚类分析"""
        print("🔬 开始工厂合约聚类分析...")
        
        # 1. 特征提取
        print("📊 提取合约特征...")
        features_df = self._extract_all_features(contracts_data)
        
        # 2. 特征预处理
        print("🔧 特征预处理和标准化...")
        X_scaled = self._preprocess_features(features_df)
        
        # 3. 确定最优聚类数
        print("🎯 确定最优聚类数...")
        optimal_k = self._find_optimal_clusters(X_scaled)
        
        # 4. 执行聚类
        print(f"🎪 执行K-Means聚类 (k={optimal_k})...")
        cluster_labels = self._perform_clustering(X_scaled, optimal_k)
        
        # 5. 聚类结果解释
        print("🔍 分析聚类结果...")
        cluster_analysis = self._analyze_clusters(features_df, cluster_labels, contracts_data)
        
        # 6. 可视化
        print("📈 生成聚类可视化...")
        self._visualize_clusters(X_scaled, cluster_labels, cluster_analysis)
        
        return {
            'features_df': features_df,
            'cluster_labels': cluster_labels,
            'cluster_analysis': cluster_analysis,
            'optimal_k': optimal_k
        }
    
    def _extract_all_features(self, contracts_data: List[Dict]) -> pd.DataFrame:
        """提取所有合约的特征"""
        all_features = []
        
        for i, contract in enumerate(contracts_data):
            if i % 50 == 0:
                print(f"  处理进度: {i}/{len(contracts_data)}")
                
            try:
                features = self.feature_extractor.extract_features(contract['source_code'])
                features['address'] = contract['address']
                all_features.append(features)
            except Exception as e:
                print(f"  警告: 处理合约 {contract['address']} 时出错: {e}")
                continue
                
        return pd.DataFrame(all_features)
    
    def _preprocess_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """特征预处理"""
        # 移除非数值特征
        numeric_features = features_df.select_dtypes(include=[np.number])
        
        # 处理缺失值
        numeric_features = numeric_features.fillna(0)
        
        # 标准化
        X_scaled = self.scaler.fit_transform(numeric_features)
        
        return X_scaled
    
    def _find_optimal_clusters(self, X: np.ndarray, max_k: int = 10) -> int:
        """使用肘部法和轮廓系数确定最优聚类数"""
        inertias = []
        silhouette_scores = []
        k_range = range(2, min(max_k + 1, len(X) // 2))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, cluster_labels))
        
        # 选择轮廓系数最高的k
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        return optimal_k
    
    def _perform_clustering(self, X: np.ndarray, k: int) -> np.ndarray:
        """执行K-Means聚类"""
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        return cluster_labels
    
    def _analyze_clusters(self, features_df: pd.DataFrame, cluster_labels: np.ndarray, 
                         contracts_data: List[Dict]) -> Dict:
        """动态分析聚类结果并自动确定类型"""
        features_df['cluster'] = cluster_labels
        
        cluster_analysis = {}
        
        for cluster_id in np.unique(cluster_labels):
            cluster_data = features_df[features_df['cluster'] == cluster_id]
            
            # 计算每个聚类的特征均值
            numeric_features = cluster_data.select_dtypes(include=[np.number])
            feature_means = numeric_features.mean()
            
            # 找出最显著的特征
            top_features = feature_means.nlargest(10)
            
            # 动态确定聚类类型
            cluster_type_info = self._determine_cluster_type_dynamically(feature_means, top_features)
            
            cluster_analysis[cluster_id] = {
                'size': len(cluster_data),
                'top_features': top_features.to_dict(),
                'cluster_type': cluster_type_info,
                'sample_addresses': cluster_data['address'].head(3).tolist()
            }
            
        return cluster_analysis
    
    def _determine_cluster_type_dynamically(self, feature_means: pd.Series, top_features: pd.Series) -> Dict:
        """基于特征动态确定聚类类型"""
        
        # 1. 技术模式识别 (基于技术特征的阈值)
        technical_patterns = {}
        if feature_means.get('clone_factory_score', 0) > 1.0:
            technical_patterns['Clone Factory'] = feature_means['clone_factory_score']
        if feature_means.get('create2_factory_score', 0) > 1.0:
            technical_patterns['CREATE2 Factory'] = feature_means['create2_factory_score']
        if feature_means.get('proxy_factory_score', 0) > 1.0:
            technical_patterns['Proxy Factory'] = feature_means['proxy_factory_score']
        if feature_means.get('registry_factory_score', 0) > 1.0:
            technical_patterns['Registry Factory'] = feature_means['registry_factory_score']
        
        # 确定主要技术模式
        primary_technical = max(technical_patterns, key=technical_patterns.get) if technical_patterns else "Standard Factory"
        
        # 2. 业务领域识别 (基于业务特征的相对强度)
        business_scores = {
            'Finance': feature_means.get('finance_score', 0),
            'NFT': feature_means.get('nft_score', 0),
            'Infrastructure': feature_means.get('infrastructure_score', 0),
            'Governance': feature_means.get('governance_score', 0)
        }
        
        # 计算业务领域的显著性 (相对于平均水平)
        total_business_score = sum(business_scores.values())
        if total_business_score > 0:
            business_distribution = {k: v/total_business_score for k, v in business_scores.items()}
            primary_business = max(business_distribution, key=business_distribution.get)
            business_confidence = business_distribution[primary_business]
        else:
            primary_business = "Generic"
            business_confidence = 0.25
        
        # 3. 特殊模式检测
        special_patterns = []
        if feature_means.get('has_create2', 0) > 0.7:
            special_patterns.append("Deterministic Deployment")
        if feature_means.get('uses_openzeppelin', 0) > 0.8:
            special_patterns.append("OpenZeppelin Based")
        if feature_means.get('total_factory_methods', 0) > 3:
            special_patterns.append("Multi-Method Factory")
        if feature_means.get('state_complexity', 0) > 5:
            special_patterns.append("Complex State Management")
        
        # 4. 动态生成类型标签
        if business_confidence > 0.6:  # 业务领域明确
            semantic_type = f"{primary_business} {primary_technical}"
        elif technical_patterns:  # 技术模式明确
            semantic_type = f"{primary_technical} Pattern"
        else:  # 通用模式
            if special_patterns:
                semantic_type = f"Generic Factory with {', '.join(special_patterns[:2])}"
            else:
                semantic_type = "Standard Factory Pattern"
        
        return {
            'semantic_type': semantic_type,
            'primary_technical': primary_technical,
            'primary_business': primary_business,
            'business_confidence': business_confidence,
            'technical_patterns': technical_patterns,
            'special_patterns': special_patterns,
            'business_distribution': business_scores
        }
    
    def _visualize_clusters(self, X: np.ndarray, cluster_labels: np.ndarray, 
                           cluster_analysis: Dict):
        """生成聚类可视化图"""
        # 使用t-SNE进行降维
        print("  执行t-SNE降维...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)//4))
        X_tsne = tsne.fit_transform(X)
        
        # 定义聚类标签和颜色
        cluster_names = []
        for cluster_id in np.unique(cluster_labels):
            domain = cluster_analysis[cluster_id]['dominant_domain']
            cluster_names.append(f"Cluster {cluster_id}: {domain.title()}")
        
        # 创建图形
        plt.figure(figsize=(12, 8))
        
        # 设置颜色
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        markers = ['o', '^', 's', 'D', 'v', '<', '>', 'p', '*', 'h']
        
        # 绘制每个聚类
        for i, cluster_id in enumerate(np.unique(cluster_labels)):
            mask = cluster_labels == cluster_id
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                       c=colors[i % len(colors)], 
                       marker=markers[i % len(markers)],
                       alpha=0.7, s=60,
                       label=cluster_names[i])
        
        plt.xlabel('Dimension 1 After Dimensionality reduction', fontsize=12)
        plt.ylabel('Dimension 2 After Dimensionality reduction', fontsize=12)
        plt.title('Clustered Semantic Space', fontsize=16, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图形
        output_path = '/Users/mac/ResearchSpace/TOSEM/experiments/RQ3/factory_clustering_result.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  聚类图已保存到: {output_path}")
        
        plt.show()

def main():
    """主函数"""
    print("🏭 工厂合约聚类分析")
    print("=" * 50)
    
    # 1. 从BigQuery获取工厂合约数据
    print("📥 从BigQuery获取工厂合约数据...")
    
    # 这里需要实现数据获取逻辑
    # contracts_data = get_factory_contracts_from_bigquery()
    
    # 为了测试，我们使用一些示例数据
    contracts_data = generate_sample_data()
    
    # 2. 执行聚类分析
    analyzer = FactoryClusteringAnalyzer()
    results = analyzer.analyze_contracts(contracts_data)
    
    # 3. 生成报告
    generate_analysis_report(results)
    
    print("✅ 聚类分析完成！")

def generate_sample_data() -> List[Dict]:
    """生成示例数据用于测试"""
    # 这里应该替换为真实的BigQuery数据获取
    return [
        {
            'address': '0x1234...', 
            'source_code': '''
            contract UniswapV2Factory {
                function createPair(address tokenA, address tokenB) external returns (address pair) {
                    // DEX factory logic
                    pair = new UniswapV2Pair();
                }
            }'''
        },
        # 添加更多示例...
    ]

def generate_analysis_report(results: Dict):
    """生成分析报告"""
    print("\n📊 聚类分析报告:")
    print("=" * 30)
    
    for cluster_id, analysis in results['cluster_analysis'].items():
        print(f"\n聚类 {cluster_id} ({analysis['dominant_domain'].title()}):")
        print(f"  合约数量: {analysis['size']}")
        print(f"  主要特征: {list(analysis['top_features'].keys())[:3]}")
        print(f"  示例地址: {analysis['sample_addresses']}")

if __name__ == "__main__":
    main()