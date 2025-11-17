"""
NVIDIA GPU Monitor with Real-Time Graph
Requires: pip install matplotlib
"""

import subprocess
import time
import threading
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

class GPUMonitor:
    def __init__(self, max_points=100):
        self.max_points = max_points
        self.times = deque(maxlen=max_points)
        self.vram_usage = deque(maxlen=max_points)
        self.gpu_util = deque(maxlen=max_points)
        self.temperatures = deque(maxlen=max_points)
        self.start_time = time.time()
        
        self.gpu_name = "Unknown"
        self.vram_total = 0
        self.max_vram = 0
        self.max_util = 0
        
        # Create figure and subplots
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(12, 8))
        self.fig.suptitle('NVIDIA GPU Monitor', fontsize=16, fontweight='bold')
        
        plt.subplots_adjust(hspace=0.4)
        
    def get_gpu_info(self):
        """Get NVIDIA GPU stats using nvidia-smi"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu', 
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                check=True
            )
            
            output = result.stdout.strip()
            if output:
                parts = output.split(',')
                self.gpu_name = parts[0].strip()
                mem_used = float(parts[1].strip())
                mem_total = float(parts[2].strip())
                gpu_util = float(parts[3].strip())
                temp = float(parts[4].strip())
                
                self.vram_total = mem_total
                self.max_vram = max(self.max_vram, mem_used)
                self.max_util = max(self.max_util, gpu_util)
                
                return mem_used, gpu_util, temp
        except Exception as e:
            print(f"Error: {e}")
            return None, None, None
    
    def update_data(self):
        """Collect GPU data"""
        mem_used, gpu_util, temp = self.get_gpu_info()
        
        if mem_used is not None:
            elapsed = time.time() - self.start_time
            self.times.append(elapsed)
            self.vram_usage.append(mem_used)
            self.gpu_util.append(gpu_util)
            self.temperatures.append(temp)
    
    def animate(self, frame):
        """Animation function for live plotting"""
        self.update_data()
        
        if len(self.times) == 0:
            return
        
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        times_list = list(self.times)
        
        # Plot 1: VRAM Usage
        self.ax1.plot(times_list, list(self.vram_usage), 'b-', linewidth=2, label='VRAM Used')
        self.ax1.axhline(y=self.vram_total, color='r', linestyle='--', linewidth=1, label=f'Total ({self.vram_total:.0f} MB)')
        self.ax1.fill_between(times_list, list(self.vram_usage), alpha=0.3)
        self.ax1.set_ylabel('VRAM (MB)', fontsize=11, fontweight='bold')
        self.ax1.set_title(f'{self.gpu_name} - VRAM Usage (Peak: {self.max_vram:.0f} MB)', fontsize=12)
        self.ax1.legend(loc='upper left')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_ylim(0, self.vram_total * 1.1)
        
        # Plot 2: GPU Utilization
        self.ax2.plot(times_list, list(self.gpu_util), 'g-', linewidth=2, label='GPU Utilization')
        self.ax2.fill_between(times_list, list(self.gpu_util), alpha=0.3, color='green')
        self.ax2.set_ylabel('GPU Utilization (%)', fontsize=11, fontweight='bold')
        self.ax2.set_title(f'GPU Utilization (Peak: {self.max_util:.0f}%)', fontsize=12)
        self.ax2.legend(loc='upper left')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_ylim(0, 100)
        
        # Plot 3: Temperature
        self.ax3.plot(times_list, list(self.temperatures), 'r-', linewidth=2, label='Temperature')
        self.ax3.fill_between(times_list, list(self.temperatures), alpha=0.3, color='red')
        self.ax3.set_ylabel('Temperature (°C)', fontsize=11, fontweight='bold')
        self.ax3.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
        self.ax3.set_title('GPU Temperature', fontsize=12)
        self.ax3.legend(loc='upper left')
        self.ax3.grid(True, alpha=0.3)
        
        # Add current values as text
        if len(times_list) > 0:
            current_vram = self.vram_usage[-1]
            current_util = self.gpu_util[-1]
            current_temp = self.temperatures[-1]
            
            self.ax1.text(0.02, 0.98, f'Current: {current_vram:.0f} MB ({current_vram/self.vram_total*100:.1f}%)', 
                         transform=self.ax1.transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            self.ax2.text(0.02, 0.98, f'Current: {current_util:.1f}%', 
                         transform=self.ax2.transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            self.ax3.text(0.02, 0.98, f'Current: {current_temp:.0f}°C', 
                         transform=self.ax3.transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    def start(self, interval=1000):
        """Start the live monitoring with graph"""
        print(f"Starting GPU monitoring for: {self.gpu_name}")
        print(f"Graph updates every {interval}ms")
        print("Close the graph window to stop monitoring.\n")
        
        ani = FuncAnimation(self.fig, self.animate, interval=interval, cache_frame_data=False)
        plt.tight_layout()
        plt.show()
        
        # Print final stats when closed
        print(f"\nMonitoring stopped.")
        print(f"Peak VRAM Usage: {self.max_vram:.0f} MB ({self.max_vram/self.vram_total*100:.1f}%)")
        print(f"Peak GPU Utilization: {self.max_util:.0f}%")

def save_to_csv(monitor, filename='gpu_log.csv'):
    """Save monitoring data to CSV"""
    import csv
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Time_Seconds', 'VRAM_MB', 'GPU_Util_%', 'Temp_C'])
        for i in range(len(monitor.times)):
            writer.writerow([
                f"{monitor.times[i]:.1f}",
                f"{monitor.vram_usage[i]:.1f}",
                f"{monitor.gpu_util[i]:.1f}",
                f"{monitor.temperatures[i]:.1f}"
            ])
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    import sys
    
    # Parse arguments
    max_points = 300  # Default: show last 5 minutes at 1s interval
    interval = 1000   # Default: update every 1000ms (1 second)
    
    if len(sys.argv) > 1:
        try:
            interval = int(float(sys.argv[1]) * 1000)  # Convert seconds to milliseconds
        except:
            print(f"Invalid interval, using default: 1 second")
    
    if len(sys.argv) > 2:
        try:
            max_points = int(sys.argv[2])
        except:
            print(f"Invalid max_points, using default: 300")
    
    print("=" * 70)
    print("NVIDIA GPU Monitor with Real-Time Graphs")
    print("=" * 70)
    print(f"Update interval: {interval/1000}s")
    print(f"Max data points: {max_points}")
    print("=" * 70)
    print()
    
    monitor = GPUMonitor(max_points=max_points)
    
    try:
        monitor.start(interval=interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
