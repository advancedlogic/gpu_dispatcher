#!/usr/bin/env python3
"""
GPU Stats Client - Example client for testing the FastAPI GPU Statistics Server
"""

import requests
import json
import time
from datetime import datetime


class GPUStatsClient:
    """Client for interacting with the FastAPI GPU Statistics Server"""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url.rstrip('/')
    
    def test_connection(self):
        """Test if the server is reachable"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def get_api_docs(self):
        """Get the OpenAPI documentation"""
        try:
            response = requests.get(f"{self.base_url}/docs")
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def get_gpu_count(self):
        """Get the number of installed GPUs"""
        try:
            response = requests.get(f"{self.base_url}/gpu/count")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting GPU count: {e}")
            return None
    
    def get_gpu_stats(self):
        """Get detailed GPU statistics"""
        try:
            response = requests.get(f"{self.base_url}/gpu/stats")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting GPU stats: {e}")
            return None
    
    def get_gpu_summary(self):
        """Get GPU usage summary"""
        try:
            response = requests.get(f"{self.base_url}/gpu/summary")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting GPU summary: {e}")
            return None
    
    def display_gpu_stats(self, stats):
        """Display GPU statistics in a readable format"""
        if not stats or "error" in stats:
            print(f"❌ Error: {stats.get('error', 'Unknown error')}")
            return
        
        print(f"\n🖥️  GPU Statistics ({stats.get('timestamp', 'Unknown time')})")
        print("=" * 60)
        print(f"📊 Total GPUs: {stats['gpu_count']}")
        
        if stats['gpu_count'] == 0:
            print("No GPUs detected")
            return
        
        for gpu in stats.get('gpus', []):
            print(f"\n🎮 GPU {gpu['gpu_id']}: {gpu['name']}")
            print(f"   💾 Memory: {gpu['memory']['used_mb']:,}MB / {gpu['memory']['total_mb']:,}MB ({gpu['memory']['usage_percent']}%)")
            print(f"   ⚡ Utilization: {gpu['utilization_percent']}%")
            
            if gpu.get('temperature_c'):
                print(f"   🌡️  Temperature: {gpu['temperature_c']}°C")
            
            if gpu.get('power', {}).get('draw_w'):
                power_draw = gpu['power']['draw_w']
                power_limit = gpu['power'].get('limit_w', 'N/A')
                print(f"   🔋 Power: {power_draw}W / {power_limit}W")
    
    def display_gpu_summary(self, summary):
        """Display GPU summary in a readable format"""
        if not summary or "error" in summary:
            print(f"❌ Error: {summary.get('error', 'Unknown error')}")
            return
        
        print(f"\n📈 GPU Summary ({summary.get('timestamp', 'Unknown time')})")
        print("=" * 50)
        print(f"🎮 Total GPUs: {summary['gpu_count']}")
        
        if summary['gpu_count'] > 0:
            print(f"💾 Total Memory: {summary['total_used_memory_mb']:,}MB / {summary['total_memory_mb']:,}MB ({summary.get('total_memory_usage_percent', 0)}%)")
            print(f"⚡ Average Utilization: {summary['average_utilization_percent']}%")
            
            print("\n📋 Per-GPU Summary:")
            for gpu in summary.get('gpus_summary', []):
                print(f"   GPU {gpu['gpu_id']} ({gpu['name'][:30]}...): {gpu['utilization_percent']}% util, {gpu['memory_usage_percent']}% mem")
    
    def monitor_gpus(self, interval=5, duration=60):
        """Monitor GPUs for a specified duration"""
        print(f"🔄 Monitoring GPUs for {duration} seconds (interval: {interval}s)")
        print("Press Ctrl+C to stop early\n")
        
        start_time = time.time()
        try:
            while time.time() - start_time < duration:
                summary = self.get_gpu_summary()
                if summary:
                    print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - ", end="")
                    if summary.get('gpu_count', 0) > 0:
                        print(f"Avg Util: {summary['average_utilization_percent']}%, "
                              f"Mem Usage: {summary.get('total_memory_usage_percent', 0)}%")
                    else:
                        print("No GPUs detected")
                
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n\n⏹️  Monitoring stopped by user")


def main():
    """Main function to demonstrate client usage"""
    client = GPUStatsClient()
    
    print("🚀 GPU Stats FastAPI Client")
    print("===========================")
    
    # Test connection
    if not client.test_connection():
        print("❌ Cannot connect to GPU server at http://localhost:5000")
        print("   Make sure the server is running: python gpu_server.py")
        print("   Or use the startup script: ./start_server.sh")
        return
    
    print("✅ Connected to FastAPI GPU server")
    
    # Check if API docs are available
    if client.get_api_docs():
        print("📚 API Documentation available at http://localhost:5000/docs")
        print("📖 Alternative docs available at http://localhost:5000/redoc")
    
    # Get and display GPU count
    print("\n1️⃣  Getting GPU count...")
    count_data = client.get_gpu_count()
    if count_data:
        print(f"   Found {count_data['gpu_count']} GPU(s)")
    
    # Get and display detailed stats
    print("\n2️⃣  Getting detailed GPU statistics...")
    stats = client.get_gpu_stats()
    client.display_gpu_stats(stats)
    
    # Get and display summary
    print("\n3️⃣  Getting GPU summary...")
    summary = client.get_gpu_summary()
    client.display_gpu_summary(summary)
    
    # Ask if user wants to monitor
    try:
        monitor = input("\n🔄 Would you like to monitor GPUs? (y/N): ").lower().strip()
        if monitor in ['y', 'yes']:
            client.monitor_gpus(interval=2, duration=30)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
