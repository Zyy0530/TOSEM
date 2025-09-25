#!/usr/bin/env python3
 

import json
import time
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContractType(Enum):
    FACTORY_CREATE = "factory_create"
    FACTORY_CREATE2 = "factory_create2"
    FACTORY_MIXED = "factory_mixed"
    FACTORY_PROXY = "factory_proxy"
    NON_FACTORY_TOKEN = "non_factory_token"
    NON_FACTORY_NFT = "non_factory_nft"
    NON_FACTORY_DEFI = "non_factory_defi"
    NON_FACTORY_GOVERNANCE = "non_factory_gov"
    NON_FACTORY_MULTISIG = "non_factory_multisig"
    NON_FACTORY_OTHER = "non_factory_other"

@dataclass
class GroundTruthContract:
    address: str
    chain: str
    is_factory: bool
    contract_type: ContractType
    bytecode: str
    created_at: datetime
    block_number: int
    tx_hash: str
    
    verification_method: str
    verified_by: str
    verification_date: datetime
    confidence_level: float
    
    contract_name: Optional[str] = None
    project_name: Optional[str] = None
    source_url: Optional[str] = None
    notes: Optional[str] = None

class GroundTruthBuilder:
    
    def __init__(self):
        self.contracts = []
        self.known_factories = self._load_known_factories()
        self.target_distribution = self._get_target_distribution()
        
    def _load_known_factories(self) -> Dict[str, Dict]:
        known_factories = {
            "ethereum": {
                "0x5c69bee701ef814a2b6a3edd4b1652cb9cc5aa6f": {
                    "name": "Uniswap V2 Factory",
                    "project": "Uniswap",
                    "type": ContractType.FACTORY_CREATE2,
                    "url": "https://github.com/Uniswap/v2-core"
                },
                "0x1f98431c8ad98523631ae4a59f267346ea31f984": {
                    "name": "Uniswap V3 Factory", 
                    "project": "Uniswap",
                    "type": ContractType.FACTORY_CREATE2,
                    "url": "https://github.com/Uniswap/v3-core"
                },
                "0xc0aee478e3658e2610c5f7a4a2e1777ce9e4f2ac": {
                    "name": "SushiSwap Factory",
                    "project": "SushiSwap", 
                    "type": ContractType.FACTORY_CREATE2,
                    "url": "https://github.com/sushiswap/sushiswap"
                },
                "0xba12222222228d8ba445958a75a0704d566bf2c8": {
                    "name": "Balancer V2 Vault",
                    "project": "Balancer",
                    "type": ContractType.FACTORY_CREATE,
                    "url": "https://github.com/balancer-labs/balancer-v2-monorepo"
                },
                "0x0959158b6040d32d04c301a72cbfd6b39e21c9ae": {
                    "name": "Curve Factory",
                    "project": "Curve",
                    "type": ContractType.FACTORY_CREATE2,
                    "url": "https://github.com/curvefi/curve-factory"
                }
            },
            "polygon": {
                "0x5757371414417b8c6caad45baef941abc7d3ab32": {
                    "name": "QuickSwap Factory",
                    "project": "QuickSwap",
                    "type": ContractType.FACTORY_CREATE2,
                    "url": "https://github.com/QuickSwap/QuickSwap-contracts"
                },
                "0xc35dadb65012ec5796536bd9864ed8773abc74c4": {
                    "name": "SushiSwap Factory (Polygon)",
                    "project": "SushiSwap",
                    "type": ContractType.FACTORY_CREATE2,
                    "url": "https://github.com/sushiswap/sushiswap"
                }
            }
        }
        return known_factories
    
    def _get_target_distribution(self) -> Dict[str, float]:
        return {
            "factory_create": 0.15,
            "factory_create2": 0.12,
            "factory_mixed": 0.05,
            "factory_proxy": 0.03,
            "non_factory_token": 0.20,
            "non_factory_nft": 0.15,
            "non_factory_defi": 0.15,
            "non_factory_governance": 0.08,
            "non_factory_multisig": 0.04,
            "non_factory_other": 0.03,
        }
    
    def add_known_factory_contracts(self):
        logger.info("Adding known factory contracts...")
        
        for chain, factories in self.known_factories.items():
            for address, info in factories.items():
                contract = GroundTruthContract(
                    address=address.lower(),
                    chain=chain,
                    is_factory=True,
                    contract_type=info["type"],
                    bytecode="",
                    created_at=datetime.now(),
                    block_number=0,
                    tx_hash="",
                    verification_method="known_project",
                    verified_by="system",
                    verification_date=datetime.now(),
                    confidence_level=1.0,
                    contract_name=info["name"],
                    project_name=info["project"],
                    source_url=info["url"],
                    notes=f"Known {info['project']} factory contract"
                )
                self.contracts.append(contract)
        
        logger.info(f"Added {len(self.contracts)} known factory contracts")
    
    def collect_random_contracts(self, chain: str, count: int) -> List[Dict]:
        logger.info(f"Collecting {count} random contracts from {chain}...")
        
        # Demo: generate mock data
        random_contracts = []
        for i in range(count):
            contract = {
                'address': f"0x{'0' * 30}{i:010x}",
                'bytecode': f"0x608060405234801561001057600080fd5b50{i:064x}",
                'created_at': datetime.now(),
                'block_number': 18000000 + i,
                'tx_hash': f"0x{'1' * 54}{i:010x}"
            }
            random_contracts.append(contract)
        
        return random_contracts
    
    def manual_annotation_interface(self, contracts: List[Dict]) -> List[GroundTruthContract]:
        logger.info("Starting manual annotation process...")
        
        annotated_contracts = []
        
        for i, contract in enumerate(contracts):
            print(f"\n=== Contract {i+1}/{len(contracts)} ===")
            print(f"Address: {contract['address']}")
            print(f"Bytecode preview: {contract['bytecode'][:100]}...")
            
            # Simplified interactive prompt
            while True:
                is_factory_input = input("Is this a factory contract? (y/n/skip): ").lower()
                if is_factory_input in ['y', 'n', 'skip']:
                    break
                print("Please enter 'y', 'n', or 'skip'")
            
            if is_factory_input == 'skip':
                continue
            
            is_factory = is_factory_input == 'y'
            
            if is_factory:
                print("Factory types:")
                print("1. CREATE factory")
                print("2. CREATE2 factory") 
                print("3. Mixed factory")
                print("4. Proxy factory")
                
                while True:
                    try:
                        type_choice = int(input("Select factory type (1-4): "))
                        if 1 <= type_choice <= 4:
                            break
                        print("Please enter a number between 1 and 4")
                    except ValueError:
                        print("Please enter a valid number")
                
                factory_types = [
                    ContractType.FACTORY_CREATE,
                    ContractType.FACTORY_CREATE2,
                    ContractType.FACTORY_MIXED,
                    ContractType.FACTORY_PROXY
                ]
                contract_type = factory_types[type_choice - 1]
            else:
                print("Non-factory types:")
                print("1. Token contract")
                print("2. NFT contract")
                print("3. DeFi application") 
                print("4. Governance contract")
                print("5. Multisig contract")
                print("6. Other")
                
                while True:
                    try:
                        type_choice = int(input("Select contract type (1-6): "))
                        if 1 <= type_choice <= 6:
                            break
                        print("Please enter a number between 1 and 6")
                    except ValueError:
                        print("Please enter a valid number")
                
                non_factory_types = [
                    ContractType.NON_FACTORY_TOKEN,
                    ContractType.NON_FACTORY_NFT,
                    ContractType.NON_FACTORY_DEFI,
                    ContractType.NON_FACTORY_GOVERNANCE,
                    ContractType.NON_FACTORY_MULTISIG,
                    ContractType.NON_FACTORY_OTHER
                ]
                contract_type = non_factory_types[type_choice - 1]
            
            confidence = float(input("Confidence level (0.0-1.0): ") or "0.8")
            notes = input("Notes (optional): ") or ""
            
            annotated_contract = GroundTruthContract(
                address=contract['address'].lower(),
                chain="ethereum",
                is_factory=is_factory,
                contract_type=contract_type,
                bytecode=contract['bytecode'],
                created_at=contract['created_at'],
                block_number=contract['block_number'],
                tx_hash=contract['tx_hash'],
                verification_method="expert_manual",
                verified_by=input("Your name: ") or "anonymous",
                verification_date=datetime.now(),
                confidence_level=confidence,
                notes=notes
            )
            
            annotated_contracts.append(annotated_contract)
        
        logger.info(f"Completed annotation of {len(annotated_contracts)} contracts")
        return annotated_contracts
    
    def validate_annotations(self, contracts: List[GroundTruthContract]) -> Dict[str, float]:
        logger.info("Validating annotation quality...")
        
        validation_results = {
            'total_contracts': len(contracts),
            'factory_ratio': sum(1 for c in contracts if c.is_factory) / len(contracts),
            'average_confidence': sum(c.confidence_level for c in contracts) / len(contracts),
            'verification_methods': {}
        }
        
        # Count verification methods
        method_counts = {}
        for contract in contracts:
            method = contract.verification_method
            method_counts[method] = method_counts.get(method, 0) + 1
        
        validation_results['verification_methods'] = {
            method: count / len(contracts) 
            for method, count in method_counts.items()
        }
        
        logger.info(f"Validation results: {validation_results}")
        return validation_results
    
    def export_dataset(self, filename: str):
        logger.info(f"Exporting dataset to {filename}...")
        
        export_data = {
            'metadata': {
                'creation_date': datetime.now().isoformat(),
                'total_contracts': len(self.contracts),
                'target_distribution': self.target_distribution,
                'validation_results': self.validate_annotations(self.contracts)
            },
            'contracts': [asdict(contract) for contract in self.contracts]
        }
        
        # Serialize datetime and Enum values
        def datetime_handler(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, ContractType):
                return obj.value
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=datetime_handler, ensure_ascii=False)
        
        logger.info(f"Dataset exported successfully: {len(self.contracts)} contracts")
    
    def load_dataset(self, filename: str):
        logger.info(f"Loading dataset from {filename}...")
        
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.contracts = []
        for contract_data in data['contracts']:
            # Convert datetime strings back to datetime objects
            contract_data['created_at'] = datetime.fromisoformat(contract_data['created_at'])
            contract_data['verification_date'] = datetime.fromisoformat(contract_data['verification_date'])
            contract_data['contract_type'] = ContractType(contract_data['contract_type'])
            
            contract = GroundTruthContract(**contract_data)
            self.contracts.append(contract)
        
        logger.info(f"Loaded {len(self.contracts)} contracts from dataset")
    
    def generate_statistics(self) -> Dict:
        stats = {
            'total_contracts': len(self.contracts),
            'factory_contracts': sum(1 for c in self.contracts if c.is_factory),
            'non_factory_contracts': sum(1 for c in self.contracts if not c.is_factory),
            'by_chain': {},
            'by_type': {},
            'by_verification_method': {},
            'confidence_distribution': {}
        }
        
        # By chain
        for contract in self.contracts:
            chain = contract.chain
            stats['by_chain'][chain] = stats['by_chain'].get(chain, 0) + 1
        
        # By type
        for contract in self.contracts:
            contract_type = contract.contract_type.value
            stats['by_type'][contract_type] = stats['by_type'].get(contract_type, 0) + 1
        
        # By verification method
        for contract in self.contracts:
            method = contract.verification_method
            stats['by_verification_method'][method] = stats['by_verification_method'].get(method, 0) + 1
        
        # Confidence distribution
        confidences = [c.confidence_level for c in self.contracts]
        stats['confidence_distribution'] = {
            'mean': sum(confidences) / len(confidences),
            'min': min(confidences),
            'max': max(confidences),
            'high_confidence': sum(1 for c in confidences if c >= 0.8) / len(confidences)
        }
        
        return stats

def main():
    print("🏗️  Factory Contract Ground Truth Dataset Builder")
    print("=" * 60)
    
    builder = GroundTruthBuilder()
    
    builder.add_known_factory_contracts()
    
    random_contracts = builder.collect_random_contracts("ethereum", 10)
    
    print("\n🏷️  Starting manual annotation process...")
    print("Note: In demo mode, automatic annotation will be used")
    
    for i, contract in enumerate(random_contracts[:3]):  # 只处理前3个作为演示
        is_factory = i % 3 == 0  # 每3个中1个是工厂合约
        contract_type = ContractType.FACTORY_CREATE if is_factory else ContractType.NON_FACTORY_TOKEN
        
        annotated_contract = GroundTruthContract(
            address=contract['address'].lower(),
            chain="ethereum",
            is_factory=is_factory,
            contract_type=contract_type,
            bytecode=contract['bytecode'],
            created_at=contract['created_at'],
            block_number=contract['block_number'],
            tx_hash=contract['tx_hash'],
            verification_method="expert_manual",
            verified_by="demo_annotator",
            verification_date=datetime.now(),
            confidence_level=0.9,
            notes="Demo annotation"
        )
        
        builder.contracts.append(annotated_contract)
    
    stats = builder.generate_statistics()
    print(f"\n📊 Dataset Statistics:")
    print(f"Total contracts: {stats['total_contracts']}")
    print(f"Factory contracts: {stats['factory_contracts']}")
    print(f"Non-factory contracts: {stats['non_factory_contracts']}")
    print(f"Factory ratio: {stats['factory_contracts']/(stats['total_contracts']):.2%}")
    print(f"Average confidence: {stats['confidence_distribution']['mean']:.3f}")
    
    dataset_filename = f"ground_truth_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    builder.export_dataset(dataset_filename)
    
    print(f"\n✅ Ground Truth dataset created successfully!")
    print(f"📁 Dataset saved to: {dataset_filename}")
    print(f"🔗 Use this dataset to evaluate your factory detector")
    
    print(f"\n🧪 Demo: Loading and using the dataset...")
    new_builder = GroundTruthBuilder()
    new_builder.load_dataset(dataset_filename)
    
    print(f"✅ Successfully loaded {len(new_builder.contracts)} contracts")
    print("🚀 Ready for detector evaluation!")

if __name__ == "__main__":
    main()
