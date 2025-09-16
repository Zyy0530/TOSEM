#!/usr/bin/env python3
"""
Final Optimized EVM CFG-based Factory Contract Detector

Final version with balanced approach between precision and recall.
Focus on reducing false negatives while maintaining good precision.
"""

import time
import json
from typing import Dict, List, Set, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from collections import deque, defaultdict


@dataclass
class FactoryResult:
    """Result of factory contract detection"""
    is_factory_contract: bool
    factory_type: str
    verified_create_positions: List[int]
    verified_create2_positions: List[int]
    false_positive_positions: List[int]
    analysis_time_ms: float
    validation_details: Dict[str, any]


class EVMInstruction(NamedTuple):
    """EVM instruction representation"""
    pc: int              # Program counter
    opcode: str         # Opcode (e.g., 'PUSH1', 'CREATE2')
    opcode_byte: str    # Raw opcode byte (e.g., '60', 'f5')
    operand: str        # Operand data (for PUSH instructions)
    operand_size: int   # Size of operand in bytes
    

class BasicBlock:
    """Basic block with conservative CREATE detection"""
    def __init__(self, start_pc: int, end_pc: int, instructions: List[EVMInstruction]):
        self.start_pc = start_pc
        self.end_pc = end_pc
        self.instructions = instructions
        self.successors: Set[int] = set()
        self.predecessors: Set[int] = set()
        self.is_reachable = False
        
    def contains_create_ops(self) -> Tuple[List[int], List[int]]:
        """Conservative CREATE/CREATE2 detection - prioritize recall over precision"""
        creates = []
        create2s = []
        
        # Check all instructions, not just those before terminators
        # This is more conservative but reduces false negatives
        for instr in self.instructions:
            if instr.opcode == 'CREATE':
                creates.append(instr.pc)
            elif instr.opcode == 'CREATE2':
                create2s.append(instr.pc)
                
        return creates, create2s


# Use the same disassembler and CFG builder from the enhanced version
class OptimizedEVMDisassembler:
    """EVM bytecode disassembler"""
    
    def __init__(self):
        # EVM opcode definitions
        self.opcodes = {
            # Arithmetic
            '00': 'STOP', '01': 'ADD', '02': 'MUL', '03': 'SUB', '04': 'DIV',
            '05': 'SDIV', '06': 'MOD', '07': 'SMOD', '08': 'ADDMOD', '09': 'MULMOD',
            '0a': 'EXP', '0b': 'SIGNEXTEND',
            
            # Comparison & Bitwise
            '10': 'LT', '11': 'GT', '12': 'SLT', '13': 'SGT', '14': 'EQ', '15': 'ISZERO',
            '16': 'AND', '17': 'OR', '18': 'XOR', '19': 'NOT', '1a': 'BYTE',
            '1b': 'SHL', '1c': 'SHR', '1d': 'SAR',
            
            # SHA3
            '20': 'SHA3',
            
            # Environmental
            '30': 'ADDRESS', '31': 'BALANCE', '32': 'ORIGIN', '33': 'CALLER',
            '34': 'CALLVALUE', '35': 'CALLDATALOAD', '36': 'CALLDATASIZE',
            '37': 'CALLDATACOPY', '38': 'CODESIZE', '39': 'CODECOPY',
            '3a': 'GASPRICE', '3b': 'EXTCODESIZE', '3c': 'EXTCODECOPY',
            '3d': 'RETURNDATASIZE', '3e': 'RETURNDATACOPY', '3f': 'EXTCODEHASH',
            
            # Block info
            '40': 'BLOCKHASH', '41': 'COINBASE', '42': 'TIMESTAMP', '43': 'NUMBER',
            '44': 'DIFFICULTY', '45': 'GASLIMIT', '46': 'CHAINID', '47': 'SELFBALANCE',
            
            # Stack, Memory, Storage
            '50': 'POP', '51': 'MLOAD', '52': 'MSTORE', '53': 'MSTORE8',
            '54': 'SLOAD', '55': 'SSTORE', '56': 'JUMP', '57': 'JUMPI',
            '58': 'PC', '59': 'MSIZE', '5a': 'GAS', '5b': 'JUMPDEST',
            
            # Push operations (PUSH1-PUSH32)
            **{f'{0x60 + i:02x}': f'PUSH{i + 1}' for i in range(32)},
            
            # Duplicate operations (DUP1-DUP16)
            **{f'{0x80 + i:02x}': f'DUP{i + 1}' for i in range(16)},
            
            # Exchange operations (SWAP1-SWAP16)
            **{f'{0x90 + i:02x}': f'SWAP{i + 1}' for i in range(16)},
            
            # Logging
            'a0': 'LOG0', 'a1': 'LOG1', 'a2': 'LOG2', 'a3': 'LOG3', 'a4': 'LOG4',
            
            # System operations
            'f0': 'CREATE', 'f1': 'CALL', 'f2': 'CALLCODE', 'f3': 'RETURN',
            'f4': 'DELEGATECALL', 'f5': 'CREATE2', 'fa': 'STATICCALL',
            'fd': 'REVERT', 'fe': 'INVALID', 'ff': 'SELFDESTRUCT'
        }
        
        # Instructions that terminate basic blocks
        self.block_terminators = {
            'JUMP', 'JUMPI', 'RETURN', 'REVERT', 'STOP', 'SELFDESTRUCT', 'INVALID'
        }

    def get_operand_size(self, opcode: str) -> int:
        """Get operand size for PUSH instructions"""
        if opcode.startswith('PUSH'):
            return int(opcode[4:])
        return 0

    def disassemble(self, bytecode: str) -> List[EVMInstruction]:
        """Disassemble bytecode into instructions"""
        if bytecode.startswith('0x'):
            bytecode = bytecode[2:]
            
        instructions = []
        pc = 0
        
        while pc * 2 < len(bytecode):
            if pc * 2 + 1 >= len(bytecode):
                break
                
            opcode_byte = bytecode[pc * 2:pc * 2 + 2].lower()
            opcode = self.opcodes.get(opcode_byte, f'UNKNOWN_{opcode_byte}')
            
            operand_size = self.get_operand_size(opcode)
            operand = ''
            
            if operand_size > 0:
                operand_start = (pc + 1) * 2
                operand_end = min(operand_start + operand_size * 2, len(bytecode))
                operand = bytecode[operand_start:operand_end]
            
            instructions.append(EVMInstruction(
                pc=pc,
                opcode=opcode,
                opcode_byte=opcode_byte,
                operand=operand,
                operand_size=operand_size
            ))
            
            pc += 1 + operand_size
            
        return instructions


class OptimizedCFGBuilder:
    """Optimized Control Flow Graph builder"""
    
    def __init__(self, disassembler: OptimizedEVMDisassembler):
        self.disasm = disassembler
        
    def identify_jump_targets(self, instructions: List[EVMInstruction]) -> Set[int]:
        """Identify all possible jump targets"""
        jump_targets = set()
        for instr in instructions:
            if instr.opcode == 'JUMPDEST':
                jump_targets.add(instr.pc)
        return jump_targets
    
    def identify_basic_block_boundaries(self, instructions: List[EVMInstruction]) -> Set[int]:
        """Identify basic block start positions"""
        boundaries = {0}
        jump_targets = self.identify_jump_targets(instructions)
        boundaries.update(jump_targets)
        
        for instr in instructions:
            if instr.opcode in self.disasm.block_terminators:
                next_pc = instr.pc + 1 + instr.operand_size
                if next_pc < len(instructions):
                    boundaries.add(next_pc)
                    
        return boundaries
    
    def build_basic_blocks(self, instructions: List[EVMInstruction]) -> Dict[int, BasicBlock]:
        """Build basic blocks from instructions"""
        if not instructions:
            return {}
            
        boundaries = sorted(self.identify_basic_block_boundaries(instructions))
        basic_blocks = {}
        
        pc_to_instr = {instr.pc: instr for instr in instructions}
        
        for i in range(len(boundaries)):
            start_pc = boundaries[i]
            end_pc = boundaries[i + 1] - 1 if i + 1 < len(boundaries) else instructions[-1].pc
            
            block_instructions = []
            pc = start_pc
            
            while pc <= end_pc and pc in pc_to_instr:
                instr = pc_to_instr[pc]
                block_instructions.append(instr)
                pc += 1 + instr.operand_size
                
            if block_instructions:
                block = BasicBlock(start_pc, end_pc, block_instructions)
                basic_blocks[start_pc] = block
                
        return basic_blocks
    
    def add_control_flow_edges(self, basic_blocks: Dict[int, BasicBlock], 
                             instructions: List[EVMInstruction]) -> None:
        """Add control flow edges - be more conservative to catch all reachable paths"""
        jump_targets = self.identify_jump_targets(instructions)
        
        for start_pc, block in basic_blocks.items():
            if not block.instructions:
                continue
                
            last_instr = block.instructions[-1]
            
            if last_instr.opcode == 'JUMP':
                # Conservative: connect to all possible jump targets
                for target in jump_targets:
                    if target in basic_blocks:
                        block.successors.add(target)
                        basic_blocks[target].predecessors.add(start_pc)
                        
            elif last_instr.opcode == 'JUMPI':
                # Fall-through path
                next_pc = last_instr.pc + 1 + last_instr.operand_size
                if next_pc in basic_blocks:
                    block.successors.add(next_pc)
                    basic_blocks[next_pc].predecessors.add(start_pc)
                
                # All possible jump targets
                for target in jump_targets:
                    if target in basic_blocks:
                        block.successors.add(target)
                        basic_blocks[target].predecessors.add(start_pc)
                
            elif last_instr.opcode in ['RETURN', 'REVERT', 'STOP', 'SELFDESTRUCT']:
                pass  # Terminal
                
            else:
                # Fall-through
                next_pc = last_instr.pc + 1 + last_instr.operand_size
                if next_pc in basic_blocks:
                    block.successors.add(next_pc)
                    basic_blocks[next_pc].predecessors.add(start_pc)


class OptimizedReachabilityAnalyzer:
    """Conservative reachability analysis"""
    
    def analyze_reachability(self, basic_blocks: Dict[int, BasicBlock], 
                           entry_point: int = 0) -> Set[int]:
        """Perform reachability analysis"""
        reachable = set()
        queue = deque([entry_point])
        
        while queue:
            current = queue.popleft()
            
            if current in reachable or current not in basic_blocks:
                continue
                
            reachable.add(current)
            basic_blocks[current].is_reachable = True
            
            for successor in basic_blocks[current].successors:
                if successor not in reachable:
                    queue.append(successor)
                    
        return reachable


class FinalFactoryDetector:
    """Final optimized factory detector with balanced precision/recall"""
    
    def __init__(self):
        self.disasm = OptimizedEVMDisassembler()
        self.cfg_builder = OptimizedCFGBuilder(self.disasm)
        self.reachability = OptimizedReachabilityAnalyzer()
        
    def detect_factory_contract(self, bytecode: str) -> FactoryResult:
        """
        Balanced factory contract detection
        """
        start_time = time.perf_counter()
        
        try:
            instructions = self.disasm.disassemble(bytecode)
            
            if not instructions:
                return FactoryResult(
                    is_factory_contract=False,
                    factory_type='NONE',
                    verified_create_positions=[],
                    verified_create2_positions=[],
                    false_positive_positions=[],
                    analysis_time_ms=(time.perf_counter() - start_time) * 1000,
                    validation_details={'error': 'No valid instructions found'}
                )
            
            # Build CFG
            basic_blocks = self.cfg_builder.build_basic_blocks(instructions)
            self.cfg_builder.add_control_flow_edges(basic_blocks, instructions)
            reachable_blocks = self.reachability.analyze_reachability(basic_blocks)
            
            # Find CREATE/CREATE2 in reachable blocks
            verified_creates = []
            verified_create2s = []
            
            for block_start in reachable_blocks:
                if block_start in basic_blocks:
                    block = basic_blocks[block_start]
                    creates, create2s = block.contains_create_ops()
                    verified_creates.extend(creates)
                    verified_create2s.extend(create2s)
            
            # Count all CREATE/CREATE2 bytes for false positive detection
            clean_bytecode = bytecode[2:] if bytecode.startswith('0x') else bytecode
            all_create_positions = []
            all_create2_positions = []
            
            for i in range(0, len(clean_bytecode) - 1, 2):
                byte_val = clean_bytecode[i:i+2].lower()
                if byte_val == 'f0':
                    all_create_positions.append(i // 2)
                elif byte_val == 'f5':
                    all_create2_positions.append(i // 2)
            
            # Calculate false positives
            false_positives = []
            for pos in all_create_positions:
                if pos not in verified_creates:
                    false_positives.append(pos)
            for pos in all_create2_positions:
                if pos not in verified_create2s:
                    false_positives.append(pos)
            
            # Determine factory type with balanced approach
            has_create = len(verified_creates) > 0
            has_create2 = len(verified_create2s) > 0
            
            # Apply a simple but effective heuristic
            # If we found any CREATE/CREATE2 in reachable code, it's likely a factory
            # But apply some basic filtering for obvious false positives
            is_factory = has_create or has_create2
            
            # Additional filtering for obvious non-factories
            if is_factory:
                # If contract is very small and has many false positives, likely not factory
                total_creates = len(all_create_positions) + len(all_create2_positions)
                if (len(instructions) < 100 and 
                    len(false_positives) >= total_creates and 
                    total_creates > 0):
                    is_factory = False
            
            if has_create and has_create2:
                factory_type = 'BOTH_CREATE_CREATE2'
            elif has_create:
                factory_type = 'CREATE_ONLY'
            elif has_create2:
                factory_type = 'CREATE2_ONLY'
            else:
                factory_type = 'NONE'
                is_factory = False
            
            analysis_time = (time.perf_counter() - start_time) * 1000
            
            return FactoryResult(
                is_factory_contract=is_factory,
                factory_type=factory_type,
                verified_create_positions=verified_creates,
                verified_create2_positions=verified_create2s,
                false_positive_positions=sorted(false_positives),
                analysis_time_ms=analysis_time,
                validation_details={
                    'total_instructions': len(instructions),
                    'total_basic_blocks': len(basic_blocks),
                    'reachable_blocks': len(reachable_blocks),
                    'all_create_bytes': len(all_create_positions),
                    'all_create2_bytes': len(all_create2_positions),
                    'final_optimized': True
                }
            )
            
        except Exception as e:
            analysis_time = (time.perf_counter() - start_time) * 1000
            return FactoryResult(
                is_factory_contract=False,
                factory_type='ERROR',
                verified_create_positions=[],
                verified_create2_positions=[],
                false_positive_positions=[],
                analysis_time_ms=analysis_time,
                validation_details={'error': str(e)}
            )
    
    def get_basic_block_info(self, bytecode: str) -> Dict:
        """Get detailed basic block information"""
        instructions = self.disasm.disassemble(bytecode)
        basic_blocks = self.cfg_builder.build_basic_blocks(instructions)
        self.cfg_builder.add_control_flow_edges(basic_blocks, instructions)
        reachable = self.reachability.analyze_reachability(basic_blocks)
        
        block_info = []
        for start_pc, block in basic_blocks.items():
            block_info.append({
                'start_pc': start_pc,
                'end_pc': block.end_pc,
                'is_reachable': start_pc in reachable,
                'successors': list(block.successors),
                'predecessors': list(block.predecessors),
                'instruction_count': len(block.instructions),
                'contains_create': any(i.opcode == 'CREATE' for i in block.instructions),
                'contains_create2': any(i.opcode == 'CREATE2' for i in block.instructions)
            })
            
        return {
            'total_instructions': len(instructions),
            'total_blocks': len(basic_blocks),
            'reachable_blocks': len(reachable),
            'blocks': block_info
        }


# Legacy compatibility functions
def detect_factory_improved(bytecode: str) -> Dict:
    """Legacy compatibility function"""
    detector = FinalFactoryDetector()
    result = detector.detect_factory_contract(bytecode)
    
    return {
        'is_factory': result.is_factory_contract,
        'factory_type': result.factory_type,
        'create_positions': result.verified_create_positions,
        'create2_positions': result.verified_create2_positions,
        'false_positives': result.false_positive_positions,
        'analysis_time_ms': result.analysis_time_ms,
        'final_optimized': True,
        'validation_details': result.validation_details
    }


# For backward compatibility, also expose as ImprovedFactoryDetector
class ImprovedFactoryDetector(FinalFactoryDetector):
    """Backward compatibility alias"""
    pass


def main():
    """Test the final detector"""
    detector = FinalFactoryDetector()
    
    print("Testing Final Optimized Factory Detector")
    print("=" * 50)
    
    try:
        with open('bytecode', 'r') as f:
            test_bytecode = f.read().strip()
            
        result = detector.detect_factory_contract(test_bytecode)
        
        print(f"Is Factory Contract: {result.is_factory_contract}")
        print(f"Factory Type: {result.factory_type}")
        print(f"CREATE positions: {len(result.verified_create_positions)}")
        print(f"CREATE2 positions: {len(result.verified_create2_positions)}")
        print(f"Analysis time: {result.analysis_time_ms:.3f}ms")
        
    except FileNotFoundError:
        print("Error: 'bytecode' file not found")
        
    print("=" * 50)


if __name__ == "__main__":
    main()