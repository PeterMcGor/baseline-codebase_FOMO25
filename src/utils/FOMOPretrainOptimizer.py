#!/usr/bin/env python3
"""
Specialized optimizer for your FOMO pretraining script.
Tests different combinations to maximize GPU utilization and training speed.
"""

import subprocess
import time
import json
import os
import re
from pathlib import Path
import pynvml as nvml

class FOMOPretrainOptimizer:
    def __init__(self, base_command_args, test_epochs=2):
        """
        base_command_args: Dictionary of your base arguments
        test_epochs: Number of epochs to run for each test (2 for quick experimentation)
        """
        self.base_args = base_command_args
        self.test_epochs = test_epochs
        self.results = []
        
        # Initialize NVIDIA ML for GPU monitoring
        try:
            nvml.nvmlInit()
            self.gpu_count = nvml.nvmlDeviceGetCount()
            print(f"🔧 Detected {self.gpu_count} GPUs")
        except:
            print("⚠️  Could not initialize NVIDIA ML - GPU monitoring disabled")
            self.gpu_count = 2  # fallback
    
    def build_command(self, config_overrides):
        """Build the training command with config overrides"""
        cmd = ["python", "src/pretrain.py"]
        
        # Start with base args
        args = self.base_args.copy()
        args.update(config_overrides)
        
        # Override epochs for testing
        args["epochs"] = self.test_epochs
        
        # Convert to command line arguments
        for key, value in args.items():
            if isinstance(value, bool) and value:
                cmd.append(f"--{key}")
            elif not isinstance(value, bool):
                cmd.append(f"--{key}={value}")
        
        return cmd
    
    def monitor_gpu_utilization(self, duration=10):
        """Monitor GPU utilization for a given duration"""
        try:
            utils = []
            memory_utils = []
            
            for _ in range(duration):
                for i in range(min(8, self.gpu_count)):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    util = nvml.nvmlDeviceGetUtilizationRates(handle)
                    memory = nvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    utils.append(util.gpu)
                    memory_utils.append((memory.used / memory.total) * 100)
                
                time.sleep(1)
            
            return {
                'avg_gpu_util': sum(utils) / len(utils),
                'avg_memory_util': sum(memory_utils) / len(memory_utils),
                'max_gpu_util': max(utils),
                'max_memory_util': max(memory_utils)
            }
        except:
            return None
    
    def run_test(self, config):
        """Run a single training test"""
        print(f"\n🔄 Testing: {config}")
        
        cmd = self.build_command(config)
        print(f"Command: {' '.join(cmd)}")
        
        try:
            start_time = time.time()
            
            # Start the process
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                env=os.environ.copy()
            )
            
            # Monitor for a bit, then let it run
            time.sleep(30)  # Let it start up
            gpu_stats = self.monitor_gpu_utilization(duration=60)  # Monitor for 1 min
            
            # Wait for completion with timeout
            try:
                stdout, stderr = process.communicate(timeout=1800)  # 30 min timeout
                end_time = time.time()
                duration = end_time - start_time
                success = process.returncode == 0
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                duration = 1800
                success = False
                print("⏰ Test timed out")
            
            # Parse throughput from logs
            throughput = self.parse_throughput(stdout)
            memory_usage = self.parse_memory_usage(stdout)
            
            result = {
                'config': config,
                'duration': duration,
                'success': success,
                'throughput_epochs_per_sec': self.test_epochs / duration if success else 0,
                'gpu_stats': gpu_stats,
                'memory_usage_gb': memory_usage,
                'stdout_tail': stdout[-2000:] if stdout else "",
                'stderr_tail': stderr[-1000:] if stderr else ""
            }
            
            self.results.append(result)
            
            if success:
                print(f"✅ Success: {duration:.1f}s")
                print(f"   Throughput: {result['throughput_epochs_per_sec']:.4f} epochs/s")
                if gpu_stats:
                    print(f"   Avg GPU Util: {gpu_stats['avg_gpu_util']:.1f}%")
                    print(f"   Avg Memory: {gpu_stats['avg_memory_util']:.1f}%")
                if memory_usage:
                    print(f"   Peak Memory: {memory_usage:.1f}GB")
            else:
                print(f"❌ Failed: {stderr[-200:] if stderr else 'Unknown error'}")
            
            return result
            
        except Exception as e:
            print(f"💥 Error: {e}")
            return {
                'config': config, 
                'success': False, 
                'error': str(e),
                'throughput_epochs_per_sec': 0
            }
    
    def parse_throughput(self, stdout):
        """Extract training speed metrics from logs"""
        # Look for Lightning's epoch timing logs
        patterns = [
            r"Epoch \d+.*?(\d+\.\d+)s",  # Epoch timing
            r"(\d+\.\d+) it/s",          # Iterations per second
            r"(\d+\.\d+)s/it"            # Seconds per iteration
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, stdout)
            if matches:
                return float(matches[-1])
        return None
    
    def parse_memory_usage(self, stdout):
        """Extract memory usage from logs"""
        pattern = r"Memory used: (\d+\.\d+) GB"
        matches = re.findall(pattern, stdout)
        if matches:
            return float(matches[-1])
        return None
    
    def test_batch_sizes(self, base_config):
        """Test different batch sizes to find the maximum"""
        print("\n🎯 Finding optimal batch size...")
        
        # Aggressive batch sizes based on patch size - user has 2x A100 80GB!
        patch_size = base_config.get('patch_size', 64)
        if patch_size == 128:
            # Aggressive testing for 128³ patches on 80GB GPUs
            batch_sizes = [32, 64, 128, 256]
        elif patch_size == 64:
            # Even more aggressive for 64³ patches
            batch_sizes = [128, 256, 512,1024]
        else:
            # Default progression
            batch_sizes = [64, 128, 256, 512]
        
        successful_batches = []
        
        for batch_size in batch_sizes:
            config = base_config.copy()
            config.update({
                'batch_size': batch_size,
                'experiment': f'batch_size_test_{batch_size}'
            })
            
            result = self.run_test(config)
            if result['success']:
                successful_batches.append((batch_size, result))
                print(f"✅ Batch size {batch_size}: {result['throughput_epochs_per_sec']:.4f} epochs/s")
                
                # Check if we're hitting memory limits
                if result['gpu_stats'] and result['gpu_stats']['max_memory_util'] > 90:
                    print(f"⚠️  High memory usage ({result['gpu_stats']['max_memory_util']:.1f}%), stopping here")
                    break
            else:
                print(f"❌ Batch size {batch_size} failed (likely OOM)")
                break
        
        if successful_batches:
            # Find the batch size with best throughput (not necessarily the largest)
            best_batch = max(successful_batches, key=lambda x: x[1]['throughput_epochs_per_sec'])
            return best_batch[0]
        return 64  # fallback
    
    def test_multi_gpu_scaling(self, base_config, optimal_batch):
        """Test scaling across multiple GPUs"""
        print(f"\n🎯 Testing multi-GPU scaling with batch_size={optimal_batch}...")
        
        # Only test 1 and 2 GPUs since user has 2 GPUs
        device_counts = [1, 2]
        scaling_results = []
        
        for num_devices in device_counts:
            if num_devices > self.gpu_count:
                continue
                
            config = base_config.copy()
            config.update({
                'num_devices': num_devices,
                'batch_size': optimal_batch,
                'experiment': f'multi_gpu_test_{num_devices}_devices'
            })
            
            result = self.run_test(config)
            if result['success']:
                scaling_results.append((num_devices, result))
                
                # Calculate scaling efficiency
                if len(scaling_results) == 1:
                    baseline_throughput = result['throughput_epochs_per_sec']
                    efficiency = 1.0
                else:
                    expected = baseline_throughput * num_devices
                    actual = result['throughput_epochs_per_sec']
                    efficiency = actual / expected
                
                print(f"✅ {num_devices} GPUs: {result['throughput_epochs_per_sec']:.4f} epochs/s (efficiency: {efficiency:.2f})")
            else:
                print(f"❌ {num_devices} GPUs failed")
        
        return scaling_results
    
    def test_worker_optimization(self, base_config):
        """Find optimal number of workers - aggressive testing for 32-core system"""
        print(f"\n🎯 Testing data loading workers (32-core system)...")
        
        # Aggressive worker testing - user has 32 cores
        worker_counts = [31]  # Focus on high-end since they have 32 cores
        best_throughput = 0
        best_workers = 31  # Default to their current setting
        
        for workers in worker_counts:
            config = base_config.copy()
            config.update({
                'num_workers': workers,
                'experiment': f'workers_test_{workers}'
            })
            
            result = self.run_test(config)
            if result['success']:
                throughput = result['throughput_epochs_per_sec']
                print(f"✅ {workers} workers: {throughput:.4f} epochs/s")
                
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_workers = workers
                    
                # Check if we're seeing diminishing returns
                if workers == 31:
                    print(f"   Using 31 workers (31 cores + 1 main process = 32 total)")
            else:
                print(f"❌ {workers} workers failed")
        
        print(f"🏆 Optimal workers: {best_workers}")
        return best_workers
    
    def test_compile_optimization(self, base_config):
        """Test compile flag and accumulation strategies"""
        print(f"\n🎯 Testing compilation and accumulation strategies...")
        
        test_configs = [
            {'compile': False, 'accumulate_grad_batches': 1, 'name': 'baseline'},
            {'compile': True, 'accumulate_grad_batches': 1, 'name': 'compile_only'},
            {'compile': False, 'accumulate_grad_batches': 2, 'name': 'accumulate_2x'},
            {'compile': True, 'accumulate_grad_batches': 2, 'name': 'compile_accumulate_2x'},
            {'compile': False, 'accumulate_grad_batches': 4, 'name': 'accumulate_4x'},
        ]
        
        results = []
        for test_config in test_configs:
            config = base_config.copy()
            config.update(test_config)
            config['experiment'] = f"optimization_test_{test_config['name']}"
            
            result = self.run_test(config)
            if result['success']:
                results.append((test_config['name'], result))
                print(f"✅ {test_config['name']}: {result['throughput_epochs_per_sec']:.4f} epochs/s")
            else:
                print(f"❌ {test_config['name']} failed")
        
        return results
    
    def test_patch_size_comparison(self, base_config):
        """Compare performance between patch sizes 64 vs 128 - AGGRESSIVE TESTING"""
        print(f"\n🎯 Comparing patch sizes 64 vs 128 (aggressive batch sizes)...")
        
        patch_configs = [
            # 64³ patches - push GPU memory hard
            #{'patch_size': 64, 'batch_size': 384, 'name': 'patch64_batch384'},
            {'patch_size': 64, 'batch_size': 512, 'name': 'patch64_batch512'}, 
            #{'patch_size': 64, 'batch_size': 640, 'name': 'patch64_batch640'},
            {'patch_size': 64, 'batch_size': 768, 'name': 'patch64_batch768'},
            
            # 128³ patches - be aggressive, user is confident GPU can handle it
            {'patch_size': 128, 'batch_size': 64, 'name': 'patch128_batch64'},
            #{'patch_size': 128, 'batch_size': 96, 'name': 'patch128_batch96'},
            {'patch_size': 128, 'batch_size': 128, 'name': 'patch128_batch128'},
            #{'patch_size': 128, 'batch_size': 160, 'name': 'patch128_batch160'},
            {'patch_size': 128, 'batch_size': 192, 'name': 'patch128_batch192'},
        ]
        
        results = []
        for patch_config in patch_configs:
            config = base_config.copy()
            config.update(patch_config)
            config['experiment'] = f"patch_test_{patch_config['name']}"
            
            result = self.run_test(config)
            if result['success']:
                results.append((patch_config['name'], result))
                throughput = result['throughput_epochs_per_sec']
                duration = result['duration']
                
                # Calculate samples/second for better comparison
                effective_batch = patch_config['batch_size'] * base_config.get('num_devices', 2)
                samples_per_sec = (effective_batch * self.test_epochs) / duration if duration > 0 else 0
                
                print(f"✅ {patch_config['name']}: {throughput:.4f} epochs/s, {samples_per_sec:.1f} samples/s")
                
                if result['gpu_stats']:
                    print(f"   GPU: {result['gpu_stats']['avg_gpu_util']:.1f}% util, {result['gpu_stats']['avg_memory_util']:.1f}% memory")
            else:
                print(f"❌ {patch_config['name']} failed - likely hit memory limit")
        
        return results
    
    def run_full_optimization(self):
        """Run the complete optimization pipeline"""
        print("🚀 Starting FOMO pretraining optimization...\n")
        
        # Base configuration for testing (2 GPU setup)
        base_config = {
            'num_devices': 2,  # User has 2 A100 80GB GPUs
            'num_workers': 31,  # User has 32 CPU cores, 31 workers + main process
            'batch_size': 64,   # Starting point
            'compile': False,
            'accumulate_grad_batches': 1,
            'patch_size': 64    # Will test both 64 and 128
        }
        
        # Step 1: Find optimal batch size
        print("="*60)
        optimal_batch = self.test_batch_sizes(base_config)
        print(f"🏆 Optimal batch size: {optimal_batch}")
        
        # Step 2: Test multi-GPU scaling
        print("="*60)
        base_config['batch_size'] = optimal_batch
        scaling_results = self.test_multi_gpu_scaling(base_config, optimal_batch)
        
        # Step 3: Optimize workers
        print("="*60)
        optimal_workers = self.test_worker_optimization(base_config)
        print(f"🏆 Optimal workers: {optimal_workers}")
        
        # Step 4: Test compilation and accumulation
        print("="*60)
        base_config['num_workers'] = optimal_workers
        optimization_results = self.test_compile_optimization(base_config)
        
        # Step 5: Compare patch sizes (64 vs 128)
        print("="*60)
        patch_size_results = self.test_patch_size_comparison(base_config)
        
        # Find the overall best configuration
        all_successful = [r for r in self.results if r['success']]
        if all_successful:
            best_result = max(all_successful, key=lambda x: x['throughput_epochs_per_sec'])
            
            print("\n" + "="*60)
            print("🏆 BEST OVERALL CONFIGURATION:")
            print("="*60)
            for key, value in best_result['config'].items():
                print(f"  --{key}={value}")
            
            print(f"\n📊 PERFORMANCE METRICS:")
            print(f"  Throughput: {best_result['throughput_epochs_per_sec']:.4f} epochs/s")
            print(f"  Duration: {best_result['duration']:.1f}s for {self.test_epochs} epochs")
            
            if best_result['gpu_stats']:
                stats = best_result['gpu_stats']
                print(f"  Avg GPU Utilization: {stats['avg_gpu_util']:.1f}%")
                print(f"  Max GPU Utilization: {stats['max_gpu_util']:.1f}%")
                print(f"  Avg Memory Usage: {stats['avg_memory_util']:.1f}%")
                print(f"  Max Memory Usage: {stats['max_memory_util']:.1f}%")
            
            if best_result['memory_usage_gb']:
                print(f"  Peak Memory: {best_result['memory_usage_gb']:.1f}GB")
            
            # Calculate time savings for full training
            baseline_time = 64 * 100 / 64  # Your current setup estimate
            optimized_time = 100 / best_result['throughput_epochs_per_sec']
            time_saved = baseline_time - optimized_time
            print(f"\n⏱️  ESTIMATED FULL TRAINING TIME:")
            print(f"  Current setup: ~{baseline_time/3600:.1f} hours")
            print(f"  Optimized setup: ~{optimized_time/3600:.1f} hours")
            print(f"  Time saved: ~{time_saved/3600:.1f} hours ({(time_saved/baseline_time)*100:.1f}% faster)")
        
        # Save results
        self.save_results()
        
        return self.results
    
    def generate_performance_report(self):
        """Generate a comprehensive performance analysis report"""
        if not self.results:
            print("No results to analyze!")
            return
        
        successful_runs = [r for r in self.results if r['success']]
        if not successful_runs:
            print("No successful runs to analyze!")
            return
        
        print("\n" + "="*80)
        print("🏆 COMPREHENSIVE PERFORMANCE REPORT")
        print("="*80)
        
        # Find best configurations by different metrics
        best_throughput = max(successful_runs, key=lambda x: x['throughput_epochs_per_sec'])
        
        # Analyze by patch size
        patch_64_runs = [r for r in successful_runs if r['config'].get('patch_size', 64) == 64]
        patch_128_runs = [r for r in successful_runs if r['config'].get('patch_size', 64) == 128]
        
        print(f"\n📊 OVERALL BEST CONFIGURATION:")
        self._print_config_details(best_throughput)
        
        if patch_64_runs:
            best_64 = max(patch_64_runs, key=lambda x: x['throughput_epochs_per_sec'])
            print(f"\n📊 BEST 64³ PATCH CONFIGURATION:")
            self._print_config_details(best_64)
        
        if patch_128_runs:
            best_128 = max(patch_128_runs, key=lambda x: x['throughput_epochs_per_sec'])
            print(f"\n📊 BEST 128³ PATCH CONFIGURATION:")
            self._print_config_details(best_128)
        
        # GPU utilization analysis
        print(f"\n⚡ GPU UTILIZATION ANALYSIS:")
        high_util_runs = [r for r in successful_runs if r.get('gpu_stats') and r['gpu_stats']['avg_gpu_util'] > 90]
        low_util_runs = [r for r in successful_runs if r.get('gpu_stats') and r['gpu_stats']['avg_gpu_util'] < 70]
        
        if high_util_runs:
            print(f"  🟢 High GPU utilization (>90%): {len(high_util_runs)} configs")
            best_high_util = max(high_util_runs, key=lambda x: x['throughput_epochs_per_sec'])
            print(f"     Best: {best_high_util['config']} → {best_high_util['gpu_stats']['avg_gpu_util']:.1f}% GPU")
        
        if low_util_runs:
            print(f"  🔴 Low GPU utilization (<70%): {len(low_util_runs)} configs")
            print(f"     → Likely data loading bottleneck or too small batch size")
        
        # Memory utilization analysis
        print(f"\n💾 MEMORY UTILIZATION ANALYSIS:")
        memory_runs = [r for r in successful_runs if r.get('gpu_stats')]
        if memory_runs:
            avg_memory = sum(r['gpu_stats']['avg_memory_util'] for r in memory_runs) / len(memory_runs)
            max_memory = max(r['gpu_stats']['max_memory_util'] for r in memory_runs)
            print(f"  Average memory usage: {avg_memory:.1f}%")
            print(f"  Maximum memory usage: {max_memory:.1f}%")
            
            if max_memory < 60:
                print(f"  🔴 Memory underutilized - you can increase batch sizes significantly!")
            elif max_memory > 90:
                print(f"  🟡 High memory usage - close to optimal")
            else:
                print(f"  🟢 Good memory utilization")
        
        # Batch size scaling analysis
        print(f"\n📈 BATCH SIZE SCALING ANALYSIS:")
        batch_analysis = {}
        for run in successful_runs:
            batch_size = run['config'].get('batch_size', 0)
            patch_size = run['config'].get('patch_size', 64)
            key = f"patch_{patch_size}"
            
            if key not in batch_analysis:
                batch_analysis[key] = []
            batch_analysis[key].append((batch_size, run['throughput_epochs_per_sec']))
        
        for patch_key, batch_data in batch_analysis.items():
            batch_data.sort()  # Sort by batch size
            print(f"  {patch_key}:")
            for batch_size, throughput in batch_data:
                print(f"    Batch {batch_size:3d}: {throughput:.4f} epochs/s")
        
        # Recommendations
        print(f"\n🎯 RECOMMENDATIONS:")
        
        # Preprocessing bottleneck detection
        gpu_bound_runs = [r for r in successful_runs if r.get('gpu_stats') and r['gpu_stats']['avg_gpu_util'] > 85]
        cpu_bound_runs = [r for r in successful_runs if r.get('gpu_stats') and r['gpu_stats']['avg_gpu_util'] < 75]
        
        if len(cpu_bound_runs) > len(gpu_bound_runs):
            print(f"  🔴 PREPROCESSING BOTTLENECK DETECTED:")
            print(f"     - Reduce num_workers or optimize data loading")
            print(f"     - Consider preprocessing data offline")
            print(f"     - Check disk I/O with 'iostat -x 1' during training")
        
        if best_throughput['config'].get('compile', False):
            print(f"  ✅ Compilation helps for your model")
        else:
            print(f"  ❌ Compilation doesn't help - UNet likely has dynamic shapes")
        
        # Time estimation for full training
        if patch_128_runs:
            best_128_throughput = max(patch_128_runs, key=lambda x: x['throughput_epochs_per_sec'])['throughput_epochs_per_sec']
            full_training_time_128 = (100 / best_128_throughput) / 3600  # hours
            print(f"  ⏱️  Estimated time for 100 epochs with 128³ patches: {full_training_time_128:.1f} hours")
        
        if patch_64_runs:
            best_64_throughput = max(patch_64_runs, key=lambda x: x['throughput_epochs_per_sec'])['throughput_epochs_per_sec']
            full_training_time_64 = (100 / best_64_throughput) / 3600  # hours
            print(f"  ⏱️  Estimated time for 100 epochs with 64³ patches: {full_training_time_64:.1f} hours")
        
        print(f"\n" + "="*80)
    
    def _print_config_details(self, result):
        """Helper to print configuration details"""
        config = result['config']
        print(f"  Configuration: {config}")
        print(f"  Throughput: {result['throughput_epochs_per_sec']:.4f} epochs/s")
        print(f"  Duration: {result['duration']:.1f}s for {self.test_epochs} epochs")
        
        if result.get('gpu_stats'):
            stats = result['gpu_stats']
            print(f"  GPU Utilization: {stats['avg_gpu_util']:.1f}% avg, {stats['max_gpu_util']:.1f}% max")
            print(f"  Memory Usage: {stats['avg_memory_util']:.1f}% avg, {stats['max_memory_util']:.1f}% max")
        
        if result.get('memory_usage_gb'):
            print(f"  Peak Memory: {result['memory_usage_gb']:.1f}GB")
    
    def save_results(self, filename="fomo_optimization_results.json"):
        """Save optimization results"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to {filename}")

# Usage
if __name__ == "__main__":
    # Add TF32 optimization for A100 GPUs
    import torch
    torch.set_float32_matmul_precision('medium')  # Speeds up A100s significantly
    
    # Your base arguments - update these paths to match your setup
    base_args = {
        "save_dir": "/home/jovyan/shared/pedro-maciasgordaliza/fomo25/FOMO-MRI/fomo-60k_baseline_preprocess_pretrain_wanb",
        "pretrain_data_dir": "/home/jovyan/shared/pedro-maciasgordaliza/fomo25/FOMO-MRI/fomo-60k_baseline_preprocess/FOMO60k",
        "model_name": "unet_b_lw_dec",
        "patch_size": 64,  # Will test both 64 and 128
        "warmup_epochs": 5,
        "augmentation_preset": "all",
        "new_version": True,
        "check_val_every_n_epoch": 5,
        "checkpoint_every_n_epochs": 5,
        "num_devices": 2,  # Your actual GPU count
        "num_workers": 31  # Your 32-core setup
    }
    
    optimizer = FOMOPretrainOptimizer(base_args, test_epochs=2)  # 2 epochs for quick experiments
    results = optimizer.run_full_optimization()