#!/usr/bin/env python3
"""
Parallel Landslide Exposure Analysis Runner for Local Machine

This script runs landslide exposure analysis in parallel on your laptop,
with configurable parallelism and progress tracking.

Usage:
    python landslide_parallel_runner.py

Configuration:
    - Edit the COUNTRIES, INFRASTRUCTURE_TYPES, and PATHS sections below
    - Adjust MAX_PARALLEL_JOBS for your system (default: 2)
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

# CRITICAL: Set multiprocessing start method to avoid deadlocks
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

# Configuration
MAX_PARALLEL_JOBS = 4

# # Countries to process (modify as needed)
# COUNTRIES = ['SVN', 'SVK', 'AUT', 'BEL', 'BGR', 'CHE', 'CYP', 
#              'CZE', 'DNK', 'EST', 'GRC',  'FIN', 'HRV', 
#              'HUN', 'IRL', 'ISL', 'LIE', 'LTU', 'LUX', 'LVA',
#              'MLT', 'NLD', 'NOR', 'PRT', 'ROU', 'SWE','DEU','ESP', 'ITA', 'POL','FRA' ] #' 

COUNTRIES = ['PRT','ESP']

# Infrastructure types to process
INFRASTRUCTURE_TYPES = [
    'telecom',
    "healthcare",
    "education",
    "power",
    "rail",
    "roads",
]

# Paths (adjust to your local setup)
LANDSLIDE_DATA_PATH = r"C:\Users\eks510\OneDrive - Vrije Universiteit Amsterdam\Documenten - MIRACA\WP3\D3.2\Hazard_data\Landslides\elsus_v2.asc"
NUTS2_PATH = r"NUTS_RG_20M_2024_3035.geojson"
SCRIPT_PATH = "landslide_exposure.py"  # Path to the main analysis script

# Output configuration
RESULTS_DIR = Path("landslide_results")
LOG_DIR = Path("logs")

def run_single_analysis(job_params):
    """
    Run a single landslide exposure analysis job.
    
    Parameters:
    -----------
    job_params : tuple
        (country_iso3, infrastructure_type, job_id, log_dir, script_path)
    
    Returns:
    --------
    dict
        Results dictionary with job status and details
    """
    country, infra_type, job_id, log_dir_str, script_path = job_params
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
            infra_type
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
        print(f"Starting {job_name}")
        
        with open(stdout_file, 'w') as out_f, open(stderr_file, 'w') as err_f:
            try:
                result = subprocess.run(
                    cmd,
                    stdout=out_f,
                    stderr=err_f,
                    text=True,
                    env=env,
                    cwd=os.getcwd()
                )
                print(f"Subprocess for {job_name} completed with return code: {result.returncode}")
            except Exception as e:
                print(f"Subprocess for {job_name} failed with exception: {e}")
                err_f.write(f"Exception during subprocess execution: {e}\n")
                raise
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Check if analysis completed successfully
        success = result.returncode == 0
        
        # Look for success indicators in stdout
        if stdout_file.exists():
            with open(stdout_file, 'r') as f:
                log_content = f.read()
                if "Results saved to" in log_content or "Analysis completed" in log_content:
                    success = True
                elif "Error" in log_content or "Failed" in log_content:
                    success = False
        
        return {
            'job_id': job_id,
            'country': country,
            'infrastructure': infra_type,
            'job_name': job_name,
            'success': success,
            'duration': duration,
            'return_code': result.returncode
        }
        
    except Exception as e:
        return {
            'job_id': job_id,
            'country': country,
            'infrastructure': infra_type,
            'job_name': job_name,
            'success': False,
            'duration': time.time() - start_time,
            'error': str(e)
        }

class LandslideRunner:
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
        """Set up simple logging configuration."""
        LOG_DIR.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
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

    def save_results_summary(self, all_results):
        """Save a comprehensive results summary."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.results_dir / f"landslide_analysis_summary_{timestamp}.json"
        
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
                'landslide_data_path': str(LANDSLIDE_DATA_PATH),
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

    def run_parallel_analysis(self):
        """Run landslide exposure analysis maintaining exactly MAX_PARALLEL_JOBS at a time."""
        print(f"Starting parallel landslide exposure analysis")
        print(f"Countries: {len(COUNTRIES)} | Infrastructure types: {len(INFRASTRUCTURE_TYPES)}")
        print(f"Max parallel jobs: {MAX_PARALLEL_JOBS}")
        
        # Validate paths
        if not Path(SCRIPT_PATH).exists():
            print(f"ERROR: Analysis script not found: {SCRIPT_PATH}")
            return False
        
        if not Path(LANDSLIDE_DATA_PATH).exists():
            print(f"ERROR: Landslide data not found: {LANDSLIDE_DATA_PATH}")
            return False
        
        # Create job combinations
        job_combinations = self.create_job_combinations()
        total_jobs = len(job_combinations)
        
        print(f"Total jobs to process: {total_jobs}")
        print("="*50)
        
        # Prepare job parameters as a queue
        from collections import deque
        job_queue = deque([
            (combo[0], combo[1], i+1, str(self.log_dir), str(SCRIPT_PATH))
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
                print(f"Started job {job_id}: {job_name}")
            
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
                                print(f"✓ Completed {result['job_name']} in {self.format_duration(result['duration'])}")
                            else:
                                self.failed_jobs.append(result)
                                error_msg = result.get('error', f"Return code: {result.get('return_code', 'unknown')}")
                                print(f"✗ Failed {result['job_name']}: {error_msg}")
                            
                        except Exception as e:
                            print(f"Error getting result for job {job_id}: {e}")
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
                        print(f"Started job {new_job_id}: {new_job_name}")
                
                # Print progress update when jobs complete
                if completed_job_ids:
                    print(f"Progress: {completed_count}/{total_jobs} ({completed_count/total_jobs*100:.1f}%) | Running: {len(running_jobs)} | Queue: {len(job_queue)}")
                
                # Sleep briefly to avoid busy waiting
                time.sleep(2)
        
        # Calculate overall statistics
        overall_end = time.time()
        overall_duration = overall_end - overall_start
        
        # Save comprehensive results
        summary_file = self.save_results_summary(all_results)
        
        # Print final summary 
        print(f"\nLANDSLIDE EXPOSURE ANALYSIS COMPLETE")
        print(f"="*50)
        print(f"Total jobs: {total_jobs}")
        print(f"Successful: {len(self.completed_jobs)}")
        print(f"Failed: {len(self.failed_jobs)}")
        print(f"Success rate: {len(self.completed_jobs)/total_jobs*100:.1f}%")
        print(f"Total runtime: {self.format_duration(overall_duration)}")
        
        if all_results:
            avg_job_time = sum(r['duration'] for r in all_results) / len(all_results)
            print(f"Average job time: {self.format_duration(avg_job_time)}")
        
        if self.failed_jobs:
            print(f"\nFailed jobs:")
            for job in self.failed_jobs:
                error_info = job.get('error', f"Return code: {job.get('return_code', 'unknown')}")
                print(f"  - {job['job_name']}: {error_info}")
        
        return len(self.failed_jobs) == 0

def main():
    """Main function to run parallel landslide exposure analysis."""
    print("Landslide Exposure Analysis - Parallel Runner")
    print(f"Max parallel jobs: {MAX_PARALLEL_JOBS}")
    
    runner = LandslideRunner()
    success = runner.run_parallel_analysis()
    
    if success:
        print("All analyses completed successfully!")
    else:
        print("Some analyses failed.")
    
    return success

if __name__ == "__main__":
    main()