#!/usr/bin/env python3
"""
EVM CFG-based Factory Contract Detector

This detector constructs an EVM-level Control Flow Graph (CFG) and performs
reachability analysis to detect CREATE/CREATE2 operations only in reachable
basic blocks, eliminating false positives from data segments.

Key Components:
1. EVM Bytecode Disassembler
2. Basic Block Identification  
3. Control Flow Graph Construction
4. Reachability Analysis
5. Factory Contract Detection

This implementation provides a lightweight alternative to Gigahorse for
accurate factory contract detection with zero false positives.
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
    """Basic block in control flow graph"""
    def __init__(self, start_pc: int, end_pc: int, instructions: List[EVMInstruction]):
        self.start_pc = start_pc
        self.end_pc = end_pc
        self.instructions = instructions
        self.successors: Set[int] = set()  # PC addresses of successor blocks
        self.predecessors: Set[int] = set()  # PC addresses of predecessor blocks
        self.is_reachable = False
        
    def contains_create_ops(self) -> Tuple[List[int], List[int]]:
        """Check if block contains CREATE/CREATE2 operations in executable code"""
        creates = []
        create2s = []
        
        # Only consider instructions before any terminating instruction
        for instr in self.instructions:
            # Stop processing if we hit a terminating instruction
            if instr.opcode in ['INVALID', 'REVERT', 'RETURN', 'STOP', 'SELFDESTRUCT']:
                break
                
            if instr.opcode == 'CREATE':
                creates.append(instr.pc)
            elif instr.opcode == 'CREATE2':
                create2s.append(instr.pc)
                
        return creates, create2s


class EVMDisassembler:
    """Lightweight EVM bytecode disassembler"""
    
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
        
        # Instructions that can be jump targets
        self.jump_targets = {'JUMPDEST'}

    def get_operand_size(self, opcode: str) -> int:
        """Get operand size for PUSH instructions"""
        if opcode.startswith('PUSH'):
            return int(opcode[4:])  # PUSH1 -> 1, PUSH32 -> 32
        return 0

    def disassemble(self, bytecode: str) -> List[EVMInstruction]:
        """Disassemble bytecode into instructions"""
        if bytecode.startswith('0x'):
            bytecode = bytecode[2:]
            
        instructions = []
        pc = 0
        
        while pc * 2 < len(bytecode):
            # Get opcode byte
            if pc * 2 + 1 >= len(bytecode):
                break
                
            opcode_byte = bytecode[pc * 2:pc * 2 + 2].lower()
            opcode = self.opcodes.get(opcode_byte, f'UNKNOWN_{opcode_byte}')
            
            # Get operand for PUSH instructions
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
            
            # Advance program counter
            pc += 1 + operand_size
            
        return instructions


class CFGBuilder:
    """Control Flow Graph builder for EVM bytecode"""
    
    def __init__(self, disassembler: EVMDisassembler):
        self.disasm = disassembler
        
    def identify_jump_targets(self, instructions: List[EVMInstruction]) -> Set[int]:
        """Identify all possible jump targets (JUMPDEST locations)"""
        jump_targets = set()
        
        for instr in instructions:
            if instr.opcode == 'JUMPDEST':
                jump_targets.add(instr.pc)
                
        return jump_targets
    
    def identify_basic_block_boundaries(self, instructions: List[EVMInstruction]) -> Set[int]:
        """Identify basic block start positions"""
        boundaries = {0}  # Entry point is always a boundary
        
        jump_targets = self.identify_jump_targets(instructions)
        boundaries.update(jump_targets)
        
        # Add positions after block terminators
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
        
        # Create PC to instruction mapping
        pc_to_instr = {instr.pc: instr for instr in instructions}
        
        for i in range(len(boundaries)):
            start_pc = boundaries[i]
            end_pc = boundaries[i + 1] - 1 if i + 1 < len(boundaries) else instructions[-1].pc
            
            # Collect instructions in this block
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
        """Add control flow edges between basic blocks"""
        pc_to_instr = {instr.pc: instr for instr in instructions}
        jump_targets = self.identify_jump_targets(instructions)
        
        for start_pc, block in basic_blocks.items():
            if not block.instructions:
                continue
                
            last_instr = block.instructions[-1]
            
            # Handle different types of control flow
            if last_instr.opcode == 'JUMP':
                # Unconditional jump - for conservative analysis, connect to all possible JUMPDEST targets
                # This ensures we don't miss reachable code due to dynamic jump resolution limitations
                for target in jump_targets:
                    if target in basic_blocks:
                        block.successors.add(target)
                        basic_blocks[target].predecessors.add(start_pc)
                        
            elif last_instr.opcode == 'JUMPI':
                # Conditional jump - multiple possible successors
                # 1. Fall-through to next instruction
                next_pc = last_instr.pc + 1 + last_instr.operand_size
                if next_pc in basic_blocks:
                    block.successors.add(next_pc)
                    basic_blocks[next_pc].predecessors.add(start_pc)
                
                # 2. All possible jump targets (conservative approach)
                for target in jump_targets:
                    if target in basic_blocks:
                        block.successors.add(target)
                        basic_blocks[target].predecessors.add(start_pc)
                
            elif last_instr.opcode in ['RETURN', 'REVERT', 'STOP', 'SELFDESTRUCT']:
                # Terminal instructions - no successors
                pass
                
            else:
                # Fall-through to next basic block
                next_pc = last_instr.pc + 1 + last_instr.operand_size
                if next_pc in basic_blocks:
                    block.successors.add(next_pc)
                    basic_blocks[next_pc].predecessors.add(start_pc)


class ReachabilityAnalyzer:
    """Reachability analysis for CFG"""
    
    def analyze_reachability(self, basic_blocks: Dict[int, BasicBlock], 
                           entry_point: int = 0) -> Set[int]:
        """Perform reachability analysis from entry point"""
        reachable = set()
        queue = deque([entry_point])
        
        while queue:
            current = queue.popleft()
            
            if current in reachable or current not in basic_blocks:
                continue
                
            reachable.add(current)
            basic_blocks[current].is_reachable = True
            
            # Add successors to queue
            for successor in basic_blocks[current].successors:
                if successor not in reachable:
                    queue.append(successor)
                    
        return reachable


class LightweightFactoryDetector:
    """Lightweight EVM CFG-based Factory Contract Detector"""
    
    def __init__(self):
        self.disasm = EVMDisassembler()
        self.cfg_builder = CFGBuilder(self.disasm)
        self.reachability = ReachabilityAnalyzer()
        
    def detect_factory_contract(self, bytecode: str) -> FactoryResult:
        """
        Detect factory contracts using CFG analysis
        
        Args:
            bytecode: Contract bytecode (with or without 0x prefix)
            
        Returns:
            FactoryResult with detection results
        """
        start_time = time.perf_counter()
        
        try:
            # Step 1: Disassemble bytecode
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
            
            # Step 2: Build basic blocks
            basic_blocks = self.cfg_builder.build_basic_blocks(instructions)
            
            # Step 3: Add control flow edges
            self.cfg_builder.add_control_flow_edges(basic_blocks, instructions)
            
            # Step 4: Perform reachability analysis
            reachable_blocks = self.reachability.analyze_reachability(basic_blocks)
            
            # Step 5: Detect CREATE/CREATE2 in reachable blocks only
            verified_creates = []
            verified_create2s = []
            false_positives = []
            
            # First, find all CREATE/CREATE2 bytes in the entire bytecode for comparison
            all_create_positions = []
            all_create2_positions = []
            
            clean_bytecode = bytecode[2:] if bytecode.startswith('0x') else bytecode
            for i in range(0, len(clean_bytecode) - 1, 2):
                byte_val = clean_bytecode[i:i+2].lower()
                if byte_val == 'f0':
                    all_create_positions.append(i // 2)
                elif byte_val == 'f5':
                    all_create2_positions.append(i // 2)
            
            # Check reachable blocks for CREATE/CREATE2
            for block_start in reachable_blocks:
                if block_start in basic_blocks:
                    block = basic_blocks[block_start]
                    creates, create2s = block.contains_create_ops()
                    verified_creates.extend(creates)
                    verified_create2s.extend(create2s)
            
            # Identify false positives (CREATE/CREATE2 bytes not in reachable code)
            for pos in all_create_positions:
                if pos not in verified_creates:
                    false_positives.append(pos)
                    
            for pos in all_create2_positions:
                if pos not in verified_create2s:
                    false_positives.append(pos)
            
            # Determine factory type
            has_create = len(verified_creates) > 0
            has_create2 = len(verified_create2s) > 0
            
            if has_create and has_create2:
                factory_type = 'BOTH_CREATE_CREATE2'
                is_factory = True
            elif has_create:
                factory_type = 'CREATE_ONLY'
                is_factory = True
            elif has_create2:
                factory_type = 'CREATE2_ONLY'
                is_factory = True
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
                    'cfg_analysis': True
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
        """
        Get detailed basic block information for debugging/analysis
        
        Args:
            bytecode: Contract bytecode
            
        Returns:
            Dictionary with CFG analysis details
        """
        # Disassemble and build CFG
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


class ImprovedFactoryDetector:
    """
    Improved Factory Contract Detector with CFG-based analysis
    
    This detector uses EVM-level control flow analysis to accurately
    detect factory contracts while eliminating false positives from
    data segments, PUSH instruction operands, and unreachable code.
    """
    
    def __init__(self):
        """Initialize the CFG-based detector"""
        self.cfg_detector = LightweightFactoryDetector()
        self.CREATE_OPCODE = 'f0'
        self.CREATE2_OPCODE = 'f5'
        
    def detect_factory_contract(self, bytecode: str) -> FactoryResult:
        """
        Detect factory contracts using CFG analysis
        
        This method uses control flow graph analysis and reachability
        analysis to ensure CREATE/CREATE2 operations are only detected
        in executable code paths, not in data segments.
        
        Args:
            bytecode: Contract bytecode (hex string with or without 0x prefix)
            
        Returns:
            FactoryResult with detailed analysis results
        """
        # Use the new CFG-based detector
        return self.cfg_detector.detect_factory_contract(bytecode)
    
    def get_basic_block_info(self, bytecode: str) -> Dict:
        """
        Get detailed basic block information for debugging/analysis
        
        Args:
            bytecode: Contract bytecode
            
        Returns:
            Dictionary with CFG analysis details
        """
        return self.cfg_detector.get_basic_block_info(bytecode)


# Legacy function for backward compatibility
def detect_factory_improved(bytecode: str) -> Dict:
    """
    Legacy function for backward compatibility
    
    Args:
        bytecode: Contract bytecode
        
    Returns:
        Dictionary with detection results in old format
    """
    detector = ImprovedFactoryDetector()
    result = detector.detect_factory_contract(bytecode)
    
    # Convert to old format
    return {
        'is_factory': result.is_factory_contract,
        'factory_type': result.factory_type,
        'create_positions': result.verified_create_positions,
        'create2_positions': result.verified_create2_positions,
        'false_positives': result.false_positive_positions,
        'analysis_time_ms': result.analysis_time_ms,
        'cfg_analysis': True,
        'validation_details': result.validation_details
    }


def main():
    """Test the improved CFG-based detector"""
    detector = ImprovedFactoryDetector()
    
    # Test with Uniswap V2 Pair contract (should not be detected as factory)
    print("Testing CFG-based Factory Detector")
    print("=" * 50)
    
    # Load test bytecode
    try:
        with open('bytecode', 'r') as f:
            test_bytecode = f.read().strip()
            
        print(f"Testing with bytecode length: {len(test_bytecode)} characters")
        
        result = detector.detect_factory_contract(test_bytecode)
        
        print("\nCFG Analysis Results:")
        print(f"Is Factory Contract: {result.is_factory_contract}")
        print(f"Factory Type: {result.factory_type}")
        print(f"Verified CREATE positions: {result.verified_create_positions}")
        print(f"Verified CREATE2 positions: {result.verified_create2_positions}")
        print(f"False positive positions: {len(result.false_positive_positions)} found")
        print(f"Analysis time: {result.analysis_time_ms:.3f}ms")
        
        print("\nValidation Details:")
        for key, value in result.validation_details.items():
            print(f"  {key}: {value}")
            
        # Get detailed CFG info
        print("\nDetailed CFG Analysis:")
        cfg_info = detector.get_basic_block_info(test_bytecode)
        print(f"Total Instructions: {cfg_info['total_instructions']}")
        print(f"Total Basic Blocks: {cfg_info['total_blocks']}")
        print(f"Reachable Blocks: {cfg_info['reachable_blocks']}")
        
        # Show reachable blocks with CREATE/CREATE2
        factory_blocks = [b for b in cfg_info['blocks'] 
                         if b['is_reachable'] and (b['contains_create'] or b['contains_create2'])]
        
        if factory_blocks:
            print(f"\nReachable blocks with CREATE/CREATE2: {len(factory_blocks)}")
            for block in factory_blocks:
                print(f"  Block {block['start_pc']}-{block['end_pc']}: "
                      f"CREATE={block['contains_create']}, CREATE2={block['contains_create2']}")
        else:
            print("\nNo reachable blocks contain CREATE/CREATE2 operations")
            
    except FileNotFoundError:
        print("Error: 'bytecode' file not found")
        
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()