# Parameter terbaik tiap keluarga distribusi

Parameter di bawah adalah hasil pendugaan kemungkinan maksimum untuk setiap keluarga, dengan parameter lokasi dikunci di nol dan karena itu tidak diduga. Urutan peringkat memakai AIC.

## Durasi sesi (`durasi_mentah`, satuan menit)

| Distribusi | Parameter terbaik keluarga itu | Lokasi |
|---|---|---|
| Log-Logistik (Fisk) | c = 1,74291; skala = 47,06317 | 0,0 (dikunci) |
| Gamma | a = 1,09633; skala = 69,56733 | 0,0 (dikunci) |
| Chi-Kuadrat | df = 2,19265; skala = 34,78370 | 0,0 (dikunci) |
| Eksponensial | skala = 76,26857 | 0,0 (dikunci) |
| Weibull | c = 0,99435; skala = 76,05769 | 0,0 (dikunci) |
| Lognormal | s = 1,10961; skala = 45,29804 | 0,0 (dikunci) |
| Half-Normal | skala = 121,93684 | 0,0 (dikunci) |
| Inverse Gaussian (Wald) | mu = 5,59200; skala = 13,63888 | 0,0 (dikunci) |
| Rayleigh | skala = 86,22236 | 0,0 (dikunci) |
| Pareto | b = 0,11935; skala = 0,01041 | 0,0 (dikunci) |

### Peringkat dan kecocokan

| Peringkat | Distribusi | Parameter bebas | Log-kemungkinan | AIC | Jarak KS | Cramer-von Mises | Anderson-Darling | Wasserstein | Massa salah tempat |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Log-Logistik (Fisk) | 2 | -118.624,2 | 237.252,5 | 0,0359 | 8,77 | 60,76 | 13,516 | 10,15 persen |
| 2 | Gamma | 2 | -119.850,0 | 239.704,0 | 0,1016 | 73,74 | 378,01 | 14,134 | 17,52 persen |
| 3 | Chi-Kuadrat | 2 | -119.850,0 | 239.704,0 | 0,1016 | 73,74 | 378,01 | 14,134 | 17,52 persen |
| 4 | Eksponensial | 1 | -119.908,9 | 239.819,7 | 0,0938 | 73,41 | 399,87 | 13,604 | 18,45 persen |
| 5 | Weibull | 2 | -119.908,1 | 239.820,3 | 0,0926 | 73,40 | 402,25 | 13,503 | 18,52 persen |
| 6 | Lognormal | 2 | -119.952,6 | 239.909,3 | 0,0810 | 41,28 | 245,43 | 11,257 | 14,50 persen |
| 7 | Half-Normal | 1 | -124.293,0 | 248.588,0 | 0,2606 | 569,08 | 2.570,39 | 32,411 | 28,40 persen |
| 8 | Inverse Gaussian (Wald) | 2 | -131.105,9 | 262.215,8 | 0,3317 | 888,43 | 4.007,45 | 32,258 | 35,17 persen |
| 9 | Rayleigh | 1 | -137.135,3 | 274.272,6 | 0,4190 | 1.688,86 | 10.508,86 | 48,613 | 45,53 persen |
| 10 | Pareto | 2 | -155.980,7 | 311.965,4 | 0,4979 | 1.714,31 | 7.966,90 | 2.235.848.108.108.306.256.691.618.792.865.792,000 | 76,38 persen |

## Jam mulai sesi (`jam_mentah_menit`, satuan jam)

| Distribusi | Parameter terbaik keluarga itu | Lokasi |
|---|---|---|
| Weibull | c = 2,76424; skala = 15,57593 | 0,0 (dikunci) |
| Rayleigh | skala = 10,59681 | 0,0 (dikunci) |
| Gamma | a = 4,02187; skala = 3,48235 | 0,0 (dikunci) |
| Chi-Kuadrat | df = 8,04371; skala = 1,74118 | 0,0 (dikunci) |
| Log-Logistik (Fisk) | c = 3,57050; skala = 13,48732 | 0,0 (dikunci) |
| Half-Normal | skala = 14,98616 | 0,0 (dikunci) |
| Lognormal | s = 0,66223; skala = 12,30509 | 0,0 (dikunci) |
| Eksponensial | skala = 14,00556 | 0,0 (dikunci) |
| Inverse Gaussian (Wald) | mu = 1,69279; skala = 8,27365 | 0,0 (dikunci) |
| Pareto | b = 0,11025; skala = 0,00142 | 0,0 (dikunci) |

### Peringkat dan kecocokan

| Peringkat | Distribusi | Parameter bebas | Log-kemungkinan | AIC | Jarak KS | Cramer-von Mises | Anderson-Darling | Wasserstein | Massa salah tempat |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Weibull | 2 | -70.692,1 | 141.388,3 | 0,0425 | 16,26 | 155,97 | 0,587 | 11,24 persen |
| 2 | Rayleigh | 1 | -72.182,2 | 144.366,3 | 0,1193 | 154,75 | 872,25 | 1,819 | 21,12 persen |
| 3 | Gamma | 2 | -73.604,2 | 147.212,3 | 0,0905 | 85,44 | 588,77 | 1,607 | 20,12 persen |
| 4 | Chi-Kuadrat | 2 | -73.604,2 | 147.212,3 | 0,0905 | 85,43 | 588,77 | 1,607 | 20,12 persen |
| 5 | Log-Logistik (Fisk) | 2 | -74.023,2 | 148.050,5 | 0,1144 | 37,79 | 417,80 | 1,896 | 19,14 persen |
| 6 | Half-Normal | 1 | -77.168,6 | 154.339,1 | 0,2762 | 656,36 | 3.176,20 | 4,080 | 38,51 persen |
| 7 | Lognormal | 2 | -79.054,4 | 158.112,8 | 0,1577 | 215,70 | 1.358,75 | 3,677 | 31,73 persen |
| 8 | Eksponensial | 1 | -81.811,3 | 163.624,6 | 0,3079 | 756,65 | 3.738,18 | 6,411 | 48,76 persen |
| 9 | Inverse Gaussian (Wald) | 2 | -92.780,3 | 185.564,6 | 0,3797 | 1.218,03 | 5.627,62 | 7,978 | 54,80 persen |
| 10 | Pareto | 2 | -128.468,6 | 256.941,2 | 0,5492 | 2.000,30 | 9.192,11 | 614.287.818.721.421.690.948.229.712.152.887.296,000 | 89,09 persen |
