# GT timeline analysis — wave_propagation_1d

Protocol rationale evidence (ADR-0032 §5). KE milestones are the
times at which the given fraction of initial kinetic energy has
dissipated; settle99 is 99% displacement settlement; tail activity
is the last-20%-of-horizon mean |acceleration| relative to its peak;
`KE diss @k` is the KE fraction dissipated within a k-frame observed
prefix — what a model at `init_frames = k` is handed for free.

| case | frames | dt | KE50 | KE90 | KE99 | settle99 | tail | KE diss @3 | KE diss @6 | KE diss @11 | peak mean aux | t(peak) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W1D-200-1 | 301 | 0.101 ms | 1.39 ms | 2.48 ms | 2.79 ms | 30 ms | 73.8% | 3.4% | 14.5% | 32.8% | 0.1367 | 8.49 ms |
| W1D-200-2 | 301 | 0.101 ms | 1.39 ms | 2.5 ms | 2.78 ms | 30 ms | 74.4% | 3.7% | 14.0% | 32.5% | 0.2721 | 8.5 ms |
| W1D-200-4 | 301 | 0.101 ms | 1.4 ms | 2.48 ms | 2.79 ms | 30 ms | 74.3% | 3.7% | 14.8% | 33.3% | 0.5377 | 8.5 ms |
| W1D-200-8 | 301 | 0.1 ms | 1.38 ms | 2.5 ms | 2.7 ms | 30 ms | 64.1% | 3.6% | 14.8% | 33.4% | 1.049 | 8.49 ms |
| W1D-300-1 | 301 | 0.101 ms | 2.1 ms | 3.8 ms | 4.18 ms | 29.8 ms | 79.7% | 2.3% | 9.6% | 21.7% | 0.1377 | 12.8 ms |
| W1D-300-2 | 301 | 0.101 ms | 2.09 ms | 3.79 ms | 4.19 ms | 29.8 ms | 76.4% | 2.4% | 9.3% | 21.5% | 0.2738 | 12.8 ms |
| W1D-300-4 | 301 | 0.101 ms | 2.08 ms | 3.69 ms | 4.09 ms | 29.8 ms | 68.9% | 2.4% | 9.8% | 22.0% | 0.5409 | 12.7 ms |
| W1D-300-8 | 301 | 0.1 ms | 2.1 ms | 3.69 ms | 4.09 ms | 29.8 ms | 57.0% | 2.4% | 9.8% | 22.1% | 1.054 | 12.7 ms |
| W1D-400-1 | 301 | 0.101 ms | 2.79 ms | 5.09 ms | 5.6 ms | 30 ms | 77.4% | 1.7% | 7.1% | 16.2% | 0.1382 | 17 ms |
| W1D-400-2 | 301 | 0.101 ms | 2.78 ms | 4.98 ms | 5.59 ms | 30 ms | 75.9% | 1.8% | 6.9% | 16.0% | 0.2749 | 17 ms |
| W1D-400-4 | 301 | 0.101 ms | 2.79 ms | 5 ms | 5.48 ms | 30 ms | 70.4% | 1.8% | 7.3% | 16.4% | 0.5428 | 17 ms |
| W1D-400-8 | 301 | 0.1 ms | 2.79 ms | 4.89 ms | 5.39 ms | 30 ms | 60.7% | 1.8% | 7.3% | 16.5% | 1.057 | 16.9 ms |
| W1D-500-1 | 301 | 0.101 ms | 3.49 ms | 6.28 ms | 6.99 ms | 30 ms | 80.4% | 1.3% | 5.7% | 13.0% | 0.1385 | 21.2 ms |
| W1D-500-2 | 301 | 0.101 ms | 3.49 ms | 6.29 ms | 6.9 ms | 30 ms | 80.0% | 1.4% | 5.5% | 12.8% | 0.2755 | 21.2 ms |
| W1D-500-4 | 301 | 0.101 ms | 3.49 ms | 6.29 ms | 6.89 ms | 30 ms | 81.6% | 1.5% | 5.8% | 13.1% | 0.5441 | 21.2 ms |
| W1D-500-8 | 301 | 0.1 ms | 3.39 ms | 6.19 ms | 6.78 ms | 30 ms | 82.4% | 1.4% | 5.8% | 13.2% | 1.058 | 21.1 ms |

## Aggregate

- Worst-case KE dissipated within candidate inits: @3: 3.7%, @6: 14.8%, @11: 33.4%
- Latest 99% settlement: 30 ms
- Peak mean aux across cases: 1.058 (latest at 21.2 ms)
