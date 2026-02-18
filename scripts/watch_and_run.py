
#!/usr/bin/env python
"""
监控进程并在其结束后自动执行命令。
用于等待数据集构建完成后自动启动训练。
"""

import argparse
import os
import time
import subprocess
import sys


def check_pid_exists(pid):
    """
    检查进程是否存在。
    
    使用 os.kill(pid, 0) - 信号0不实际发送信号，
    但会进行错误检查，可用于检测进程是否存在。
    """
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main():
    parser = argparse.ArgumentParser(description='Watch a process and run a command after it finishes')
    parser.add_argument('--pid', type=int, required=True, help='Process ID to watch')
    parser.add_argument('--cmd', type=str, required=True, help='Command to run after process finishes')
    parser.add_argument('--check_interval', type=int, default=60, 
                        help='Check interval in seconds (default: 60)')
    args = parser.parse_args()
    
    print(f"WatchAndRun started")
    print(f"Watching PID: {args.pid}")
    print(f"Command to run: {args.cmd}")
    print(f"Check interval: {args.check_interval} seconds")
    print("-" * 50)
    
    try:
        # 监控循环
        while True:
            if check_pid_exists(args.pid):
                current_time = time.strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{current_time}] Waiting for PID {args.pid} to finish...")
                time.sleep(args.check_interval)
            else:
                break
        
        # 进程已结束
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')
        print("-" * 50)
        print(f"[{current_time}] PID {args.pid} has finished!")
        print(f"Executing command: {args.cmd}")
        print("-" * 50)
        
        # 执行命令
        result = subprocess.run(args.cmd, shell=True)
        
        if result.returncode == 0:
            print(f"Command executed successfully with return code {result.returncode}")
            sys.exit(0)
        else:
            print(f"Command failed with return code {result.returncode}")
            sys.exit(result.returncode)
            
    except KeyboardInterrupt:
        print("\nWatchAndRun interrupted by user")
        sys.exit(130)


if __name__ == "__main__":
    main()

