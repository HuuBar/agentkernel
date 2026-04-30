"""Generate benchmark report"""
import subprocess
import sys

def run_all():
    benchmarks = [
        "benchmarks/01_openai_vs_bacg.py",
        "benchmarks/02_cascade_control.py",
        "benchmarks/03_multi_agent_budget.py",
    ]

    results = []
    for bench in benchmarks:
        print(f"\nRunning {bench}...")
        result = subprocess.run([sys.executable, bench], capture_output=True, text=True)
        results.append({"name": bench, "output": result.stdout, "error": result.stderr})

    print("\n" + "=" * 60)
    print("BENCHMARK REPORT")
    print("=" * 60)
    for r in results:
        print(f"\n--- {r['name']} ---")
        print(r['output'][-500:] if len(r['output']) > 500 else r['output'])
        if r['error']:
            print(f"STDERR: {r['error']}")

    return results

if __name__ == "__main__":
    run_all()
