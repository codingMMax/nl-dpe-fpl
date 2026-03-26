# Attention Head Benchmark: NL-DPE vs Azure-Lily vs Baseline FPGA

Head dimension d = 128, crossbar 512×128

## Energy + Latency Comparison

| N | | Energy (nJ) | Latency (µs) | E vs BL | L vs BL |
|---|---|---|---|---|---|
| 128 | NL-DPE | 2,918.8 | 114.9 | 3.27× | 3.64× |
| 128 | Azure-Lily | 4,745.5 | 252.8 | 2.01× | 1.65× |
| 128 | Baseline | 9,542.0 | 418.1 | 1.00× | 1.00× |
| | | | | | |
| 256 | NL-DPE | 10,962.4 | 283.4 | 2.44× | 3.10× |
| 256 | Azure-Lily | 17,141.2 | 588.2 | 1.56× | 1.49× |
| 256 | Baseline | 26,734.2 | 878.9 | 1.00× | 1.00× |
| | | | | | |
| 512 | NL-DPE | 42,205.2 | 778.9 | 1.99× | 3.06× |
| 512 | Azure-Lily | 64,886.6 | 1,842.6 | 1.30× | 1.29× |
| 512 | Baseline | 84,072.5 | 2,384.2 | 1.00× | 1.00× |
| | | | | | |
| 1024 | NL-DPE | 165,095.5 | 2,399.9 | 1.76× | 3.08× |
| 1024 | Azure-Lily | 252,189.8 | 6,349.9 | 1.15× | 1.16× |
| 1024 | Baseline | 290,561.7 | 7,393.3 | 1.00× | 1.00× |
| | | | | | |

## Key Findings

- NL-DPE wins on BOTH energy and latency vs Azure-Lily and Baseline
- N=128: 1.6× energy / 2.2× latency vs AL, 3.3× / 3.6× vs BL
- N=256: 1.6× energy / 2.1× latency vs AL, 2.4× / 3.1× vs BL
- N=512: 1.5× energy / 2.4× latency vs AL, 2.0× / 3.1× vs BL
- N=1024: 1.5× energy / 2.6× latency vs AL, 1.8× / 3.1× vs BL
