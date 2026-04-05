#!/usr/bin/env python3
"""
execBench.py — benchmark de funções VGL via execWorkflowGen.py

Uso:
  python3 execBench.py <workflow.wksp> [--n N] [--device GPU|CPU|BOTH] [--csv out.csv]

Exemplos:
  python3 execBench.py data/workflows/fundus/fundus.wksp --n 100 --device GPU
  python3 execBench.py data/workflows/fundus/fundus.wksp --n 100 --device BOTH --csv bench.csv
"""
import argparse
import subprocess
import sys
import re
import csv
import os


def run_bench(workflow, n, device):
    """Executa execWorkflowGen.py e retorna dict {func: ms}."""
    print(f"  Rodando {n} iterações [{device}]...", flush=True)
    result = subprocess.run(
        [sys.executable, "execWorkflowGen.py", workflow, str(n), device],
        capture_output=True, text=True,
        env=os.environ.copy(),
    )
    timings = {}
    for line in result.stdout.splitlines():
        m = re.match(r'\[BENCH\]\s+(.+?):\s+([\d.]+)\s+ms', line)
        if m:
            func, ms = m.group(1), float(m.group(2))
            if func != 'TOTAL':
                timings[func] = ms
    if not timings:
        print(f"  Aviso: nenhuma linha [BENCH] capturada. Stderr:", file=sys.stderr)
        for line in result.stderr.splitlines()[-5:]:
            print(f"    {line}", file=sys.stderr)
    return timings


def print_table(results, n):
    devices = list(results.keys())
    all_funcs = list(dict.fromkeys(f for d in devices for f in results[d]))
    n_ops = len(all_funcs)

    col_w = 16
    width = 38 + col_w * len(devices)
    col_header = "".join(f"{f'{d} (ms)':>{col_w}}" for d in devices)
    header = f"{'Operação':<38}" + col_header

    print(f"\nBenchmark — {n} iterações × {n_ops} operações")
    print("─" * width)
    print(header)
    print("─" * width)

    totals = {d: 0.0 for d in devices}
    for func in all_funcs:
        row = f"{func:<38}"
        for d in devices:
            val = results[d].get(func, 0.0)
            totals[d] += val
            row += f"{val:>{col_w}.3f}"
        print(row)

    print("─" * width)
    total_row = f"{'TOTAL':<38}"
    for d in devices:
        total_row += f"{totals[d]:>{col_w}.3f}"
    print(total_row)
    print("─" * width)


def save_csv(results, n, path):
    devices = list(results.keys())
    all_funcs = list(dict.fromkeys(f for d in devices for f in results[d]))

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Operação'] + [f'{d} (ms)' for d in devices])
        totals = {d: 0.0 for d in devices}
        for func in all_funcs:
            row = [func]
            for d in devices:
                val = results[d].get(func, 0.0)
                totals[d] += val
                row.append(f'{val:.3f}')
            writer.writerow(row)
        writer.writerow(['TOTAL'] + [f'{totals[d]:.3f}' for d in devices])

    print(f"\nSalvo em: {path}")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark de funções VGL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('workflow', help='Arquivo .wksp')
    parser.add_argument('--n', type=int, default=100,
                        help='Número de iterações por função (padrão: 100)')
    parser.add_argument('--device', default='GPU', choices=['GPU', 'CPU', 'BOTH'],
                        help='Device a usar (padrão: GPU)')
    parser.add_argument('--csv', metavar='FILE',
                        help='Salvar resultado em CSV')
    args = parser.parse_args()

    devices = ['GPU', 'CPU'] if args.device == 'BOTH' else [args.device]
    results = {}
    for d in devices:
        results[d] = run_bench(args.workflow, args.n, d)

    print_table(results, args.n)

    if args.csv:
        save_csv(results, args.n, args.csv)


if __name__ == '__main__':
    main()
