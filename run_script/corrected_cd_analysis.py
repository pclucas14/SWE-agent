#!/usr/bin/env python3

import json
import sys
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

class CorrectedCDAnalyzer:
    """修正后的CD分析器 - 只考虑助手的连续bash命令"""
    
    def __init__(self):
        pass
        
    def _parse_function_call_from_content(self, content: str) -> List[str]:
        """从助手消息内容解析函数调用"""
        actions = []
        
        if not content or not isinstance(content, str):
            return []
        
        # 查找所有函数调用
        function_pattern = r'<function=([^>]+)>(.*?)</function>'
        function_matches = re.findall(function_pattern, content, re.DOTALL)
        
        for function_name, function_content in function_matches:
            if function_name == 'bash':
                # 提取bash命令参数
                parameter_pattern = r'<parameter=([^>]+)>(.*?)</parameter>'
                parameter_matches = re.findall(parameter_pattern, function_content, re.DOTALL)
                
                for param_name, param_value in parameter_matches:
                    if param_name == 'command':
                        actions.append(param_value.strip())
                        break
                else:
                    actions.append('bash_no_command')
        
        return actions
    
    def analyze_assistant_bash_sequences(self, file_path: str):
        """只分析助手的bash命令序列"""
        
        print(f"\n=== 修正后的CD分析: {Path(file_path).name} ===")
        
        try:
            trajectories = []
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        trajectories = data
                    else:
                        trajectories = [data]
                except json.JSONDecodeError:
                    f.seek(0)
                    lines = f.readlines()
                    if lines:
                        for line in lines:
                            if line.strip():
                                trajectories.append(json.loads(line))
                    else:
                        print("空文件")
                        return None
        except Exception as e:
            print(f"加载文件错误: {e}")
            return None
        
        print(f"找到 {len(trajectories)} 个轨迹进行分析")
        
        # 统计数据
        total_bash_commands = []
        consecutive_cd_pairs = []
        cd_sequences = []
        all_assistant_actions = []  # 包括非bash的助手动作
        
        for traj_idx, trajectory in enumerate(trajectories):
            if (traj_idx + 1) % 500 == 0 or traj_idx < 10:
                print(f"处理轨迹 {traj_idx + 1}/{len(trajectories)}")
            
            # 处理不同数据结构
            messages = trajectory.get('messages', [])
            if not messages and 'history' in trajectory:
                messages = trajectory.get('history', [])
            
            # 提取助手的所有动作（按顺序）
            assistant_bash_sequence = []
            assistant_all_actions = []
            
            for message in messages:
                if message.get("role") == "assistant":
                    content = message.get("content", "")
                    
                    # 检查是否有函数调用
                    function_pattern = r'<function=([^>]+)>'
                    function_matches = re.findall(function_pattern, content)
                    
                    if function_matches:
                        for func_name in function_matches:
                            assistant_all_actions.append(func_name)
                            
                            if func_name == 'bash':
                                # 提取bash命令
                                bash_commands = self._parse_function_call_from_content(content)
                                for bash_cmd in bash_commands:
                                    assistant_bash_sequence.append(bash_cmd)
                                    total_bash_commands.append(bash_cmd)
            
            all_assistant_actions.extend(assistant_all_actions)
            
            # 分析这个轨迹中的连续cd模式和所有连续相同命令
            current_cd_sequence = []
            prev_bash_cmd = None
            
            for i, bash_cmd in enumerate(assistant_bash_sequence):
                if bash_cmd.strip().startswith('cd '):
                    current_cd_sequence.append(bash_cmd.strip())
                    
                    # 检查是否与前一个bash命令构成连续的cd对
                    if prev_bash_cmd and prev_bash_cmd.strip().startswith('cd '):
                        consecutive_cd_pairs.append((prev_bash_cmd.strip(), bash_cmd.strip()))
                else:
                    # 非cd命令，结束当前cd序列
                    if len(current_cd_sequence) > 1:
                        cd_sequences.append({
                            'trajectory': traj_idx,
                            'sequence': current_cd_sequence.copy(),
                            'length': len(current_cd_sequence)
                        })
                    current_cd_sequence = []
                
                prev_bash_cmd = bash_cmd
            
            # 处理轨迹末尾的cd序列
            if len(current_cd_sequence) > 1:
                cd_sequences.append({
                    'trajectory': traj_idx,
                    'sequence': current_cd_sequence.copy(),
                    'length': len(current_cd_sequence)
                })
        
        # 统计结果
        total_cd_commands = sum(1 for cmd in total_bash_commands if cmd.strip().startswith('cd '))
        cd_targets = [cmd.strip()[3:].strip() for cmd in total_bash_commands 
                     if cmd.strip().startswith('cd ') and len(cmd.strip()) > 3]
        unique_cd_targets = len(set(cd_targets))
        
        # 计算所有连续的相同bash命令（真正的冗余命令）
        all_consecutive_same_commands = []
        for i in range(len(total_bash_commands) - 1):
            cmd1 = total_bash_commands[i].strip()
            cmd2 = total_bash_commands[i + 1].strip()
            if cmd1 == cmd2:
                all_consecutive_same_commands.append((cmd1, cmd2))
        
        # 冗余命令数量（所有相同的连续bash命令）
        redundant_commands_count = len(all_consecutive_same_commands)
        
        print(f"\n=== 修正后的分析结果 ===")
        print(f"总bash命令: {len(total_bash_commands)}")
        print(f"总cd命令: {total_cd_commands}")
        print(f"连续cd对: {len(consecutive_cd_pairs)}")
        print(f"冗余bash命令: {redundant_commands_count}")
        print(f"唯一cd目标: {unique_cd_targets}")
        print(f"cd序列(长度>1): {len(cd_sequences)}")
        
        if total_cd_commands > 0:
            print(f"连续cd比率: {len(consecutive_cd_pairs) / total_cd_commands:.2%}")
            print(f"效率比率: {unique_cd_targets / total_cd_commands:.2%}")
            print(f"cd命令百分比: {total_cd_commands / len(total_bash_commands):.2%}")
        
        # 显示连续cd对的例子
        if consecutive_cd_pairs:
            print(f"\n=== 连续CD对示例 (前10个) ===")
            cd_counter = Counter(consecutive_cd_pairs)
            for i, ((cd1, cd2), count) in enumerate(cd_counter.most_common(10)):
                print(f"  {i+1:2d}. {count:3d}x: {cd1} → {cd2}")
        
        # 显示冗余bash命令的例子
        if all_consecutive_same_commands:
            print(f"\n=== 冗余Bash命令示例 (前10个) ===")
            redundant_counter = Counter(all_consecutive_same_commands)
            for i, ((cmd1, cmd2), count) in enumerate(redundant_counter.most_common(10)):
                cmd_preview = cmd1[:60] + "..." if len(cmd1) > 60 else cmd1
                print(f"  {i+1:2d}. {count:3d}x: {cmd_preview}")
        
        # 显示cd序列长度分布
        if cd_sequences:
            print(f"\n=== CD序列长度分布 ===")
            length_dist = Counter(seq['length'] for seq in cd_sequences)
            for length in sorted(length_dist.keys()):
                print(f"  长度 {length}: {length_dist[length]} 个序列")
            
            # 显示最长的序列示例
            longest_sequences = sorted(cd_sequences, key=lambda x: x['length'], reverse=True)[:3]
            print(f"\n=== 最长CD序列示例 ===")
            for i, seq in enumerate(longest_sequences):
                print(f"  {i+1}. 轨迹 {seq['trajectory']}, 长度 {seq['length']}")
                print(f"     前3个命令: {seq['sequence'][:3]}")
                if len(seq['sequence']) > 3:
                    print(f"     后3个命令: {seq['sequence'][-3:]}")
                print()
        
        return {
            'total_bash': len(total_bash_commands),
            'total_cd': total_cd_commands,
            'consecutive_pairs': len(consecutive_cd_pairs),
            'redundant': redundant_commands_count,
            'unique_targets': unique_cd_targets,
            'cd_sequences': len(cd_sequences),
            'consecutive_cd_pairs': consecutive_cd_pairs,
            'cd_sequences_detailed': cd_sequences
        }


def main():
    analyzer = CorrectedCDAnalyzer()
    
    # 分析两个数据文件
    files_to_analyze = [
        {
            "name": "Claude Sonnet-4 (Astropy)",
            "path": "/home/zhengyanshi/project/SWE-agent/data/automated_pipeline_o3_bugs30_combos50_depth2_workers32_nbugs1_patches2_perfile2_permodule10/swesmith_gen_claude__claude-sonnet-4__t-0.00__p-1.00__c-2.00___patch_swesmith_astropy__astropy.26d14786_ps__ms50_as1/astropy__astropy.26d14786_submit.json"
        },
        {
            "name": "SWE-Smith Trajectories", 
            "path": "/home/zhengyanshi/project/SWE-agent/data/swe-smith-trajectories.jsonl"
        }
    ]
    
    results = {}
    
    for file_info in files_to_analyze:
        print(f"\n{'='*80}")
        print(f"分析: {file_info['name']}")
        print(f"{'='*80}")
        
        result = analyzer.analyze_assistant_bash_sequences(file_info['path'])
        if result:
            results[file_info['name']] = result
    
    # 比较分析
    if len(results) > 1:
        print(f"\n{'='*80}")
        print("修正后的比较分析")
        print(f"{'='*80}")
        
        claude_result = results.get("Claude Sonnet-4 (Astropy)", {})
        smith_result = results.get("SWE-Smith Trajectories", {})
        
        print(f"\n{'指标':<25} {'Claude Sonnet-4':<15} {'SWE-Smith':<15} {'差异'}")
        print("-" * 65)
        
        metrics = [
            ("总Bash命令", 'total_bash'),
            ("总CD命令", 'total_cd'),
            ("连续CD对", 'consecutive_pairs'),
            ("冗余命令", 'redundant'),
            ("唯一目标", 'unique_targets'),
            ("CD序列", 'cd_sequences')
        ]
        
        for metric_name, key in metrics:
            claude_val = claude_result.get(key, 0)
            smith_val = smith_result.get(key, 0)
            diff = claude_val - smith_val
            print(f"{metric_name:<25} {claude_val:<15} {smith_val:<15} {diff:+}")
        
        print(f"\n{'比率指标'}")
        print("-" * 65)
        for name, result in results.items():
            if result['total_cd'] > 0:
                consecutive_ratio = result['consecutive_pairs'] / result['total_cd']
                efficiency_ratio = result['unique_targets'] / result['total_cd']
                cd_percentage = result['total_cd'] / result['total_bash']
                
                print(f"\n{name}:")
                print(f"  连续CD比率: {consecutive_ratio:.2%}")
                print(f"  效率比率:   {efficiency_ratio:.2%}")
                print(f"  CD命令占比: {cd_percentage:.2%}")


if __name__ == "__main__":
    main()