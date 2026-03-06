#!/usr/bin/env python3
"""
Parallel Wildfire Exposure Analysis Runner for Local Machine

This script runs wildfire exposure analysis in parallel on your laptop,
with configurable parallelism and progress tracking.

Usage:
    python wildfire_parallel_runner.py

Configuration:
    - Edit the COUNTRIES, INFRASTRUCTURE_TYPES, and PATHS sections below
    - Adjust MAX_PARALLEL_JOBS for your system (default: 4)
"""

import subprocess
import multiprocessing as mp
import time
import json
from pathlib import Path
from datetime import datetime
import sys
import os
import logging
import tempfile
import shutil

# CRITICAL: Set multiprocessing start method to avoid deadlocks
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

# Configuration
MAX_PARALLEL_JOBS = 2

# Countries to process (modify as needed)
COUNTRIES = ['SVN', 'SVK', 'AUT',  'BEL', 'BGR', 'CHE', 'CYP', 
             'CZE', 'DEU', 'DNK', 'EST', 'GRC', 'ESP', 'FIN', 'FRA', 'HRV', 
             'HUN', 'IRL', 'ISL', 'ITA', 'LIE', 'LTU', 'LUX', 'LVA',
             'MLT', 'NLD', 'NOR', 'POL', 'PRT', 'ROU', 'SWE']

COUNTRIES = ['PRT','ESP']

# Infrastructure types to process
INFRASTRUCTURE_TYPES = [
    "healthcare",
    "education",
    "power",
    "rail",
    "roads", 
]

# Paths (adjust to your local setup)
WILDFIRE_DATA_PATH = r"C:\Users\eks510\OneDrive - Vrije Universiteit Amsterdam\Documenten - MIRACA\WP3\D3.2\Hazard_data\Wildfires"
NUTS2_PATH = r"NUTS_RG_20M_2024_3035.geojson"
SCRIPT_PATH = "wildfire_exposure.py"  # Path to the main analysis script

# Output configuration
RESULTS_DIR = Path("wildfire_results")
LOG_DIR = Path("logs")

def get_job_progress(stdout_file):
    """Get the current progress of a job by reading its stdout file."""
    progress_keywords = [
        ("Starting", "Starting analysis..."),
        ("Loading wildfire data", "Loading wildfire data"),
        ("Processing infrastructure", "Processing infrastructure"),
        ("Calculating exposure", "Calculating exposure"),
        ("Processing NUTS2", "Processing NUTS2 regions"),
        ("Saving results", "Saving results"),
        ("Analysis completed", "Completed successfully")
    ]
    
    if not stdout_file.exists():
        return "Starting..."
    
    try:
        with open(stdout_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().lower()
            
            # Check progress keywords in reverse order (most advanced first)
            for keyword, status in reversed(progress_keywords):
                if keyword.lower() in content:
                    return status
                    
            # Check for errors
            if any(error_word in content for error_word in ['error', 'failed', 'exception']):
                return "Error detected"
                
    except Exception:
        pass
    
    return "Running..."

def run_single_analysis(job_params):
    """
    Run a single wildfire analysis job.
    
    Parameters:
    -----------
    job_params : tuple
        (country_iso3, infrastructure_type, job_id, log_dir, script_path, wildfire_path, nuts2_path)
    
    Returns:
    --------
    dict
        Results dictionary with job status and details
    """
    country, infra_type, job_id, log_dir_str, script_path, wildfire_path, nuts2_path = job_params
    job_name = f"{country}_{infra_type}"
    start_time = time.time()
    log_dir = Path(log_dir_str)
    
    try:
        # Create command with unbuffered Python
        cmd = [
            sys.executable,
            '-u',  # Force unbuffered stdout/stderr
            script_path,
            country,
            infra_type,
            str(wildfire_path),
            str(nuts2_path)
        ]
        
        # Create log files for this job
        stdout_file = log_dir / f"{job_name}_stdout.log"
        stderr_file = log_dir / f"{job_name}_stderr.log"
        
        # Set environment variables to reduce conflicts and force unbuffered output
        env = os.environ.copy()
        env['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
        env['GDAL_CACHEMAX'] = '256'
        env['PYTHONUNBUFFERED'] = '1'  # Force unbuffered output
        
        # Run the subprocess
        print(f"[DEBUG] Starting subprocess for {job_name} with command: {' '.join(cmd)}")
        
        with open(stdout_file, 'w') as out_f, open(stderr_file, 'w') as err_f:
            # Write initial debug info
            out_f.write(f"=== STARTING JOB {job_name} ===\n")
            out_f.write(f"Command: {' '.join(cmd)}\n")
            out_f.write(f"Working directory: {os.getcwd()}\n")
            out_f.write(f"Python executable: {sys.executable}\n")
            out_f.flush()
            
            try:
                result = subprocess.run(
                    cmd,
                    stdout=out_f,
                    stderr=err_f,
                    text=True,
                    timeout=360000,
                    env=env,
                    cwd=os.getcwd()  # Explicitly set working directory
                )
                print(f"[DEBUG] Subprocess for {job_name} completed with return code: {result.returncode}")
            except Exception as e:
                print(f"[DEBUG] Subprocess for {job_name} failed with exception: {e}")
                err_f.write(f"Exception during subprocess execution: {e}\n")
                raise
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Check if analysis completed successfully
        success = result.returncode == 0
        
        # Look for success indicators in stdout and get more info
        if stdout_file.exists():
            with open(stdout_file, 'r') as f:
                log_content = f.read()
                print(f"[DEBUG] {job_name} stdout content ({len(log_content)} chars):")
                print(f"[DEBUG] First 500 chars: {log_content[:500]}")
                if "Analysis completed successfully!" in log_content:
                    success = True
                elif "Error" in log_content or "Failed" in log_content:
                    success = False
                elif len(log_content) < 100:  # Very short output suggests early exit
                    print(f"[DEBUG] WARNING: {job_name} has very short output, may have failed silently")
        
        # Also check stderr
        if stderr_file.exists():
            with open(stderr_file, 'r') as f:
                err_content = f.read()
                if err_content.strip():
                    print(f"[DEBUG] {job_name} stderr content: {err_content[:200]}")
        
        return {
            'job_id': job_id,
            'country': country,
            'infrastructure': infra_type,
            'job_name': job_name,
            'success': success,
            'duration': duration,
            'return_code': result.returncode,
            'start_time': start_time,
            'end_time': end_time,
            'stdout_file': str(stdout_file),
            'stderr_file': str(stderr_file)
        }
        
    except subprocess.TimeoutExpired:
        return {
            'job_id': job_id,
            'country': country,
            'infrastructure': infra_type,
            'job_name': job_name,
            'success': False,
            'duration': time.time() - start_time,
            'error': 'Timeout (>1 hour)',
            'start_time': start_time,
            'end_time': time.time()
        }
        
    except Exception as e:
        return {
            'job_id': job_id,
            'country': country,
            'infrastructure': infra_type,
            'job_name': job_name,
            'success': False,
            'duration': time.time() - start_time,
            'error': str(e),
            'start_time': start_time,
            'end_time': time.time()
        }

class WildfireRunner:
    def __init__(self):
        self.setup_logging()
        self.results_dir = RESULTS_DIR
        self.log_dir = LOG_DIR
        self.results_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # Track job status
        self.completed_jobs = []
        self.failed_jobs = []
        self.active_jobs = {}  # Store currently running jobs info
            
    def setup_logging(self):
        """Set up logging configuration."""
        LOG_DIR.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(LOG_DIR / 'wildfire_parallel.log', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def create_job_combinations(self):
        """Create all combinations of countries and infrastructure types."""
        combinations = []
        for country in COUNTRIES:
            for infra_type in INFRASTRUCTURE_TYPES:
                combinations.append((country, infra_type))
        return combinations

    def format_duration(self, seconds):
        """Format duration in human-readable format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def print_progress_summary(self, completed, total, pool=None):
        """Print a progress summary with running job details."""
        current_time = time.time()
        
        # Only update every N seconds to avoid spam
        if current_time - self.last_progress_update < self.progress_update_interval:
            return
        
        self.last_progress_update = current_time
        
        print(f"\n{'='*80}")
        print(f"PROGRESS UPDATE: {completed}/{total} completed ({completed/total*100:.1f}%)")
        print(f"Successful: {len(self.completed_jobs)} | Failed: {len(self.failed_jobs)}")
        
        # Show currently running jobs with their progress
        print(f"\nCurrently running jobs:")
        print("-" * 80)
        
        if self.active_jobs:
            for job_id, job_info in self.active_jobs.items():
                job_name = job_info['job_name']
                start_time = job_info['start_time']
                stdout_file = self.log_dir / f"{job_name}_stdout.log"
                
                runtime = current_time - start_time
                progress = get_job_progress(stdout_file)
                
                runtime_str = self.format_duration(runtime)
                print(f"  Job {job_id:2d}: {job_name:<15} | Runtime: {runtime_str} | Status: {progress}")
        else:
            print("  No jobs currently running.")
        
        print(f"{'='*80}\n")

    def save_results_summary(self, all_results):
        """Save a comprehensive results summary."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.results_dir / f"wildfire_analysis_summary_{timestamp}.json"
        
        # Create summary statistics
        total_jobs = len(all_results)
        successful_jobs = sum(1 for r in all_results if r['success'])
        failed_jobs = total_jobs - successful_jobs
        
        total_duration = sum(r['duration'] for r in all_results)
        avg_duration = total_duration / total_jobs if total_jobs > 0 else 0
        
        summary = {
            'timestamp': timestamp,
            'configuration': {
                'max_parallel_jobs': MAX_PARALLEL_JOBS,
                'countries': COUNTRIES,
                'infrastructure_types': INFRASTRUCTURE_TYPES,
                'wildfire_data_path': str(WILDFIRE_DATA_PATH),
                'nuts2_path': str(NUTS2_PATH),
                'multiprocessing_method': 'spawn'
            },
            'statistics': {
                'total_jobs': total_jobs,
                'successful_jobs': successful_jobs,
                'failed_jobs': failed_jobs,
                'success_rate': successful_jobs / total_jobs * 100 if total_jobs > 0 else 0,
                'total_duration_seconds': total_duration,
                'total_duration_formatted': self.format_duration(total_duration),
                'average_job_duration_seconds': avg_duration,
                'average_job_duration_formatted': self.format_duration(avg_duration)
            },
            'job_results': all_results
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary_file

    def job_callback(self, result):
        """Callback function when a job completes successfully."""
        if result['success']:
            self.completed_jobs.append(result)
            self.logger.info(f"[SUCCESS] Completed {result['job_name']} in {self.format_duration(result['duration'])}")
        else:
            self.failed_jobs.append(result)
            error_msg = result.get('error', f"Return code: {result.get('return_code', 'unknown')}")
            self.logger.error(f"[FAILED] {result['job_name']}: {error_msg}")
        
        # Remove from active jobs
        if result['job_id'] in self.active_jobs:
            del self.active_jobs[result['job_id']]

    def job_error_callback(self, error):
        """Callback function when a job encounters an error."""
        self.logger.error(f"Job encountered an error: {error}")

    def run_parallel_analysis(self):
        """Run wildfire analysis maintaining exactly MAX_PARALLEL_JOBS at a time."""
        self.logger.info(f"Starting parallel wildfire exposure analysis")
        self.logger.info(f"Multiprocessing method: spawn (Windows optimized)")
        self.logger.info(f"Countries: {COUNTRIES}")
        self.logger.info(f"Infrastructure types: {INFRASTRUCTURE_TYPES}")
        self.logger.info(f"Max parallel jobs: {MAX_PARALLEL_JOBS}")
        
        # Validate paths
        if not Path(SCRIPT_PATH).exists():
            self.logger.error(f"Analysis script not found: {SCRIPT_PATH}")
            return False
        
        if not Path(WILDFIRE_DATA_PATH).exists():
            self.logger.error(f"Wildfire data path not found: {WILDFIRE_DATA_PATH}")
            return False
        
        if not Path(NUTS2_PATH).exists():
            self.logger.error(f"NUTS2 path not found: {NUTS2_PATH}")
            return False
        
        # Create job combinations
        job_combinations = self.create_job_combinations()
        total_jobs = len(job_combinations)
        
        self.logger.info(f"Total jobs to process: {total_jobs}")
        
        # Prepare job parameters as a queue
        from collections import deque
        job_queue = deque([
            (combo[0], combo[1], i+1, str(self.log_dir), str(SCRIPT_PATH), 
             str(WILDFIRE_DATA_PATH), str(NUTS2_PATH))
            for i, combo in enumerate(job_combinations)
        ])
        
        # Track overall timing
        overall_start = time.time()
        all_results = []
        completed_count = 0
        
        # Use multiprocessing Pool but control job submission
        with mp.Pool(processes=MAX_PARALLEL_JOBS) as pool:
            running_jobs = {}  # job_id -> (async_result, job_info)
            
            # Start initial batch of jobs (up to MAX_PARALLEL_JOBS)
            while len(running_jobs) < MAX_PARALLEL_JOBS and job_queue:
                job_params = job_queue.popleft()
                job_id = job_params[2]
                job_name = f"{job_params[0]}_{job_params[1]}"
                
                # Track this job as active
                self.active_jobs[job_id] = {
                    'job_name': job_name,
                    'start_time': time.time()
                }
                
                # Submit job to pool
                async_result = pool.apply_async(
                    run_single_analysis, 
                    args=(job_params,)
                )
                
                running_jobs[job_id] = (async_result, self.active_jobs[job_id])
                self.logger.info(f"Started job {job_id}: {job_name}")
            
            # Monitor and manage jobs
            while running_jobs or job_queue:
                # Check for completed jobs
                completed_job_ids = []
                
                for job_id, (async_result, job_info) in running_jobs.items():
                    if async_result.ready():
                        try:
                            result = async_result.get()
                            all_results.append(result)
                            completed_count += 1
                            completed_job_ids.append(job_id)
                            
                            # Log completion
                            if result['success']:
                                self.completed_jobs.append(result)
                                self.logger.info(f"[SUCCESS] Completed {result['job_name']} in {self.format_duration(result['duration'])}")
                            else:
                                self.failed_jobs.append(result)
                                error_msg = result.get('error', f"Return code: {result.get('return_code', 'unknown')}")
                                self.logger.error(f"[FAILED] {result['job_name']}: {error_msg}")
                            
                        except Exception as e:
                            self.logger.error(f"Error getting result for job {job_id}: {e}")
                            completed_count += 1
                            completed_job_ids.append(job_id)
                
                # Remove completed jobs and start new ones
                for job_id in completed_job_ids:
                    del running_jobs[job_id]
                    if job_id in self.active_jobs:
                        del self.active_jobs[job_id]
                    
                    # Start a new job if there are jobs waiting
                    if job_queue:
                        job_params = job_queue.popleft()
                        new_job_id = job_params[2]
                        new_job_name = f"{job_params[0]}_{job_params[1]}"
                        
                        # Track this job as active
                        self.active_jobs[new_job_id] = {
                            'job_name': new_job_name,
                            'start_time': time.time()
                        }
                        
                        # Submit new job to pool
                        async_result = pool.apply_async(
                            run_single_analysis, 
                            args=(job_params,)
                        )
                        
                        running_jobs[new_job_id] = (async_result, self.active_jobs[new_job_id])
                        self.logger.info(f"Started job {new_job_id}: {new_job_name}")
                
                # Print progress update only when jobs complete or start (not on timer)
                if completed_job_ids:
                    print(f"\n{'='*60}")
                    print(f"JOBS COMPLETED: {len(completed_job_ids)}")
                    print(f"TOTAL PROGRESS: {completed_count}/{total_jobs} ({completed_count/total_jobs*100:.1f}%)")
                    print(f"CURRENTLY RUNNING: {len(running_jobs)} jobs")
                    print(f"JOBS IN QUEUE: {len(job_queue)}")
                    print(f"{'='*60}\n")
                
                # Sleep briefly to avoid busy waiting
                time.sleep(1)
        
        # Calculate overall statistics
        overall_end = time.time()
        overall_duration = overall_end - overall_start
        
        # Save comprehensive results
        summary_file = self.save_results_summary(all_results)
        
        # Print final summary
        print(f"\n{'='*80}")
        print(f"WILDFIRE EXPOSURE ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"Total jobs: {total_jobs}")
        print(f"Successful: {len(self.completed_jobs)}")
        print(f"Failed: {len(self.failed_jobs)}")
        print(f"Success rate: {len(self.completed_jobs)/total_jobs*100:.1f}%")
        print(f"Total runtime: {self.format_duration(overall_duration)}")
        
        if all_results:
            avg_job_time = sum(r['duration'] for r in all_results) / len(all_results)
            print(f"Average job time: {self.format_duration(avg_job_time)}")
        
        print(f"Results summary saved to: {summary_file}")
        
        if self.failed_jobs:
            print(f"\nFailed jobs:")
            for job in self.failed_jobs:
                error_info = job.get('error', f"Return code: {job.get('return_code', 'unknown')}")
                print(f"  - {job['job_name']}: {error_info}")
                if 'stderr_file' in job:
                    print(f"    Error log: {job['stderr_file']}")
        
        print(f"{'='*80}")
        
        return len(self.failed_jobs) == 0

def main():
    """Main function to run parallel wildfire analysis."""
    print("Wildfire Exposure Analysis - Parallel Runner (Pool-based)")
    print(f"Using multiprocessing.Pool with 'spawn' method")
    print(f"Starting analysis with up to {MAX_PARALLEL_JOBS} parallel jobs...")
    
    runner = WildfireRunner()
    success = runner.run_parallel_analysis()
    
    if success:
        print("All analyses completed successfully!")
        sys.exit(0)
    else:
        print("Some analyses failed. Check the logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()