#!/usr/bin/env python3
"""
工厂合约检测器有效性评估脚本
使用Ground Truth数据集评估检测器的Precision、Recall等指标
"""

import json
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# 导入我们的检测器和Ground Truth构建器
from factory_detector import ImprovedFactoryDetector
from ground_truth_builder import GroundTruthBuilder, GroundTruthContract, ContractType

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """单个合约的评估结果"""
    address: str
    ground_truth_label: bool
    predicted_label: bool
    factory_type_gt: Optional[str]
    factory_type_pred: Optional[str]
    confidence: Optional[float]
    analysis_time_ms: float
    is_correct: bool

@dataclass
class EvaluationMetrics:
    """评估指标"""
    # 基本指标
    true_positives: int
    false_positives: int
    true_negatives: int 
    false_negatives: int
    
    # 派生指标
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    specificity: float
    
    # 性能指标
    avg_analysis_time_ms: float
    total_analysis_time_ms: float

class DetectorEvaluator:
    """检测器评估器"""
    
    def __init__(self, detector=None, ground_truth_dataset=None):
        """初始化评估器"""
        self.detector = detector or ImprovedFactoryDetector()
        self.ground_truth_dataset = ground_truth_dataset or []
        self.evaluation_results = []
        self.metrics = None
        
    def load_ground_truth_dataset(self, dataset_file: str):
        """加载Ground Truth数据集"""
        logger.info(f"Loading ground truth dataset from {dataset_file}")
        
        builder = GroundTruthBuilder()
        builder.load_dataset(dataset_file)
        self.ground_truth_dataset = builder.contracts
        
        logger.info(f"Loaded {len(self.ground_truth_dataset)} contracts for evaluation")
    
    def run_evaluation(self) -> List[EvaluationResult]:
        """运行完整评估"""
        logger.info("Starting detector evaluation...")
        
        self.evaluation_results = []
        total_contracts = len(self.ground_truth_dataset)
        
        for i, contract in enumerate(self.ground_truth_dataset):
            logger.info(f"Evaluating contract {i+1}/{total_contracts}: {contract.address}")
            
            # 运行检测器
            start_time = time.perf_counter()
            try:
                detection_result = self.detector.detect_factory_contract(contract.bytecode)
                analysis_time = (time.perf_counter() - start_time) * 1000  # 转换为毫秒
                
                predicted_label = detection_result.is_factory_contract
                factory_type_pred = detection_result.factory_type if predicted_label else None
                confidence = getattr(detection_result, 'confidence', None)
                
            except Exception as e:
                logger.error(f"Error analyzing contract {contract.address}: {e}")
                analysis_time = (time.perf_counter() - start_time) * 1000
                predicted_label = False
                factory_type_pred = None
                confidence = None
            
            # 创建评估结果
            result = EvaluationResult(
                address=contract.address,
                ground_truth_label=contract.is_factory,
                predicted_label=predicted_label,
                factory_type_gt=contract.contract_type.value if contract.is_factory else None,
                factory_type_pred=factory_type_pred,
                confidence=confidence,
                analysis_time_ms=analysis_time,
                is_correct=(contract.is_factory == predicted_label)
            )
            
            self.evaluation_results.append(result)
            
            # 每50个合约报告一次进度
            if (i + 1) % 50 == 0:
                accuracy_so_far = sum(1 for r in self.evaluation_results if r.is_correct) / len(self.evaluation_results)
                logger.info(f"Progress: {i+1}/{total_contracts}, Current accuracy: {accuracy_so_far:.3f}")
        
        logger.info("Evaluation completed!")
        return self.evaluation_results
    
    def calculate_metrics(self) -> EvaluationMetrics:
        """计算评估指标"""
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run evaluation first.")
        
        logger.info("Calculating evaluation metrics...")
        
        # 统计TP, FP, TN, FN
        tp = sum(1 for r in self.evaluation_results 
                if r.ground_truth_label == True and r.predicted_label == True)
        fp = sum(1 for r in self.evaluation_results 
                if r.ground_truth_label == False and r.predicted_label == True)
        tn = sum(1 for r in self.evaluation_results 
                if r.ground_truth_label == False and r.predicted_label == False)
        fn = sum(1 for r in self.evaluation_results 
                if r.ground_truth_label == True and r.predicted_label == False)
        
        # 计算派生指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(self.evaluation_results)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # 性能指标
        analysis_times = [r.analysis_time_ms for r in self.evaluation_results]
        avg_analysis_time = np.mean(analysis_times)
        total_analysis_time = sum(analysis_times)
        
        self.metrics = EvaluationMetrics(
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            accuracy=accuracy,
            specificity=specificity,
            avg_analysis_time_ms=avg_analysis_time,
            total_analysis_time_ms=total_analysis_time
        )
        
        return self.metrics
    
    def print_metrics_report(self):
        """打印详细的指标报告"""
        if not self.metrics:
            self.calculate_metrics()
        
        print("\n" + "="*80)
        print("🏭 FACTORY CONTRACT DETECTOR EVALUATION REPORT")
        print("="*80)
        
        # 基本统计
        print(f"\n📊 BASIC STATISTICS:")
        print(f"Total contracts evaluated: {len(self.evaluation_results)}")
        print(f"Ground truth factories: {self.metrics.true_positives + self.metrics.false_negatives}")
        print(f"Ground truth non-factories: {self.metrics.true_negatives + self.metrics.false_positives}")
        
        # 混淆矩阵
        print(f"\n🎯 CONFUSION MATRIX:")
        print(f"                 Predicted")
        print(f"                 Factory  Non-Factory")
        print(f"Actual Factory    {self.metrics.true_positives:4d}      {self.metrics.false_negatives:4d}")
        print(f"    Non-Factory   {self.metrics.false_positives:4d}      {self.metrics.true_negatives:4d}")
        
        # 性能指标
        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"Precision:    {self.metrics.precision:.4f}")
        print(f"Recall:       {self.metrics.recall:.4f}")
        print(f"F1-Score:     {self.metrics.f1_score:.4f}")
        print(f"Accuracy:     {self.metrics.accuracy:.4f}")
        print(f"Specificity:  {self.metrics.specificity:.4f}")
        
        # 性能时间
        print(f"\n⏱️  TIME PERFORMANCE:")
        print(f"Average analysis time: {self.metrics.avg_analysis_time_ms:.2f} ms")
        print(f"Total analysis time:   {self.metrics.total_analysis_time_ms/1000:.2f} seconds")
        print(f"Contracts per second:  {len(self.evaluation_results)/(self.metrics.total_analysis_time_ms/1000):.2f}")
    
    def analyze_errors(self) -> Dict[str, List[EvaluationResult]]:
        """分析错误案例"""
        logger.info("Analyzing error cases...")
        
        false_positives = [r for r in self.evaluation_results 
                          if not r.ground_truth_label and r.predicted_label]
        false_negatives = [r for r in self.evaluation_results 
                          if r.ground_truth_label and not r.predicted_label]
        
        print(f"\n❌ ERROR ANALYSIS:")
        print(f"False Positives: {len(false_positives)}")
        print(f"False Negatives: {len(false_negatives)}")
        
        if false_positives:
            print(f"\n🔍 FALSE POSITIVE EXAMPLES:")
            for i, fp in enumerate(false_positives[:5]):  # 显示前5个
                print(f"  {i+1}. {fp.address} (predicted as {fp.factory_type_pred})")
        
        if false_negatives:
            print(f"\n🔍 FALSE NEGATIVE EXAMPLES:")
            for i, fn in enumerate(false_negatives[:5]):  # 显示前5个
                print(f"  {i+1}. {fn.address} (actual type: {fn.factory_type_gt})")
        
        return {
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def analyze_by_contract_type(self):
        """按合约类型分析性能"""
        logger.info("Analyzing performance by contract type...")
        
        # 按Ground Truth类型分组
        type_performance = {}
        
        for result in self.evaluation_results:
            gt_contract = next(c for c in self.ground_truth_dataset if c.address == result.address)
            contract_type = gt_contract.contract_type.value
            
            if contract_type not in type_performance:
                type_performance[contract_type] = {
                    'total': 0,
                    'correct': 0,
                    'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0
                }
            
            stats = type_performance[contract_type]
            stats['total'] += 1
            
            if result.is_correct:
                stats['correct'] += 1
            
            # 更新混淆矩阵统计
            if result.ground_truth_label and result.predicted_label:
                stats['tp'] += 1
            elif not result.ground_truth_label and result.predicted_label:
                stats['fp'] += 1
            elif not result.ground_truth_label and not result.predicted_label:
                stats['tn'] += 1
            else:  # ground_truth_label and not result.predicted_label
                stats['fn'] += 1
        
        print(f"\n📋 PERFORMANCE BY CONTRACT TYPE:")
        print(f"{'Type':<25} {'Total':<6} {'Accuracy':<10} {'Precision':<10} {'Recall':<8}")
        print("-" * 65)
        
        for contract_type, stats in sorted(type_performance.items()):
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
            recall = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
            
            print(f"{contract_type:<25} {stats['total']:<6} {accuracy:<10.3f} {precision:<10.3f} {recall:<8.3f}")
        
        return type_performance
    
    def generate_visualizations(self, output_dir: str = "./evaluation_plots"):
        """生成可视化图表"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Generating visualizations in {output_dir}...")
        
        # 设置图表样式
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. 混淆矩阵热力图
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        y_true = [r.ground_truth_label for r in self.evaluation_results]
        y_pred = [r.predicted_label for r in self.evaluation_results]
        
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Non-Factory', 'Factory'],
                    yticklabels=['Non-Factory', 'Factory'], ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 性能指标雷达图
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        metrics_names = ['Precision', 'Recall', 'F1-Score', 'Accuracy', 'Specificity']
        metrics_values = [
            self.metrics.precision,
            self.metrics.recall, 
            self.metrics.f1_score,
            self.metrics.accuracy,
            self.metrics.specificity
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
        metrics_values += metrics_values[:1]  # 闭合多边形
        angles += angles[:1]
        
        ax.plot(angles, metrics_values, 'o-', linewidth=2, color='blue')
        ax.fill(angles, metrics_values, alpha=0.25, color='blue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_names)
        ax.set_ylim(0, 1)
        ax.set_title('Performance Metrics Radar Chart')
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_radar.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 分析时间分布直方图
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        analysis_times = [r.analysis_time_ms for r in self.evaluation_results]
        ax.hist(analysis_times, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax.set_xlabel('Analysis Time (ms)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Contract Analysis Times')
        ax.axvline(np.mean(analysis_times), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(analysis_times):.2f} ms')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/analysis_time_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualizations saved to {output_dir}/")
    
    def export_detailed_results(self, filename: str):
        """导出详细的评估结果"""
        logger.info(f"Exporting detailed results to {filename}")
        
        # 准备导出数据
        export_data = {
            'metadata': {
                'evaluation_date': datetime.now().isoformat(),
                'total_contracts': len(self.evaluation_results),
                'detector_version': getattr(self.detector, 'version', 'unknown')
            },
            'metrics': {
                'precision': self.metrics.precision,
                'recall': self.metrics.recall,
                'f1_score': self.metrics.f1_score,
                'accuracy': self.metrics.accuracy,
                'specificity': self.metrics.specificity,
                'confusion_matrix': {
                    'tp': self.metrics.true_positives,
                    'fp': self.metrics.false_positives,
                    'tn': self.metrics.true_negatives,
                    'fn': self.metrics.false_negatives
                },
                'performance': {
                    'avg_analysis_time_ms': self.metrics.avg_analysis_time_ms,
                    'total_analysis_time_ms': self.metrics.total_analysis_time_ms
                }
            },
            'detailed_results': [
                {
                    'address': r.address,
                    'ground_truth': r.ground_truth_label,
                    'predicted': r.predicted_label,
                    'gt_factory_type': r.factory_type_gt,
                    'pred_factory_type': r.factory_type_pred,
                    'confidence': r.confidence,
                    'analysis_time_ms': r.analysis_time_ms,
                    'correct': r.is_correct
                }
                for r in self.evaluation_results
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Detailed results exported to {filename}")

def main():
    """主函数 - 演示评估流程"""
    print("🧪 Factory Contract Detector Evaluation")
    print("=" * 50)
    
    # 创建评估器
    evaluator = DetectorEvaluator()
    
    # 模拟：加载Ground Truth数据集
    # 实际使用时，请替换为真实的数据集文件
    print("📁 Loading ground truth dataset...")
    # evaluator.load_ground_truth_dataset("ground_truth_dataset_20241209_123456.json")
    
    # 演示模式：创建模拟数据集
    from ground_truth_builder import GroundTruthContract, ContractType
    demo_contracts = []
    
    # 添加一些模拟的工厂合约
    factory_bytecode = "0x608060405234801561001057600080fd5b5061012345678901234567890123456789012345678901234567890123456789abcdef"
    demo_contracts.append(GroundTruthContract(
        address="0x1111111111111111111111111111111111111111",
        chain="ethereum",
        is_factory=True,
        contract_type=ContractType.FACTORY_CREATE,
        bytecode=factory_bytecode,
        created_at=datetime.now(),
        block_number=18000000,
        tx_hash="0xaaaa",
        verification_method="known_project",
        verified_by="demo",
        verification_date=datetime.now(),
        confidence_level=1.0
    ))
    
    # 添加一些模拟的非工厂合约
    token_bytecode = "0x608060405234801561001057600080fd5b50fedcba0987654321abcdef1234567890abcdef1234567890abcdef1234567890"
    demo_contracts.append(GroundTruthContract(
        address="0x2222222222222222222222222222222222222222", 
        chain="ethereum",
        is_factory=False,
        contract_type=ContractType.NON_FACTORY_TOKEN,
        bytecode=token_bytecode,
        created_at=datetime.now(),
        block_number=18000001,
        tx_hash="0xbbbb",
        verification_method="expert_manual",
        verified_by="demo",
        verification_date=datetime.now(),
        confidence_level=0.9
    ))
    
    evaluator.ground_truth_dataset = demo_contracts
    
    print(f"✅ Loaded {len(evaluator.ground_truth_dataset)} contracts for evaluation")
    
    # 运行评估
    print("\n🚀 Running detector evaluation...")
    evaluator.run_evaluation()
    
    # 计算和显示指标
    print("\n📊 Calculating metrics...")
    evaluator.calculate_metrics()
    evaluator.print_metrics_report()
    
    # 错误分析
    evaluator.analyze_errors()
    
    # 按类型分析
    evaluator.analyze_by_contract_type()
    
    # 生成可视化（需要安装matplotlib和seaborn）
    try:
        evaluator.generate_visualizations()
    except ImportError:
        print("⚠️  Visualization libraries not available. Skipping plots.")
    
    # 导出结果
    results_file = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    evaluator.export_detailed_results(results_file)
    
    print(f"\n🎉 Evaluation completed!")
    print(f"📋 Summary:")
    print(f"   - Precision: {evaluator.metrics.precision:.4f}")
    print(f"   - Recall:    {evaluator.metrics.recall:.4f}")
    print(f"   - F1-Score:  {evaluator.metrics.f1_score:.4f}")
    print(f"   - Accuracy:  {evaluator.metrics.accuracy:.4f}")
    print(f"\n📁 Results saved to: {results_file}")

if __name__ == "__main__":
    main()