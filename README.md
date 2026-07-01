# Trabalho Prático 3 - Otimização Paralela de GEMM

Este repositório reúne o material desenvolvido para um trabalho prático sobre otimização de multiplicação de matrizes densas (GEMM), com foco em desempenho em CPU e GPU. O projeto compara implementações sequenciais e otimizadas, incluindo versões com blocagem, paralelismo via OpenMP e aceleração via CUDA, além de registrar resultados experimentais em arquivos CSV e consolidá-los em um notebook e em um relatório técnico.

## Estrutura dos arquivos

- `Trabalho Prático 3.ipynb`: notebook principal com a implementação, os experimentos e a parte final do relatório.
- `gemm_cpu.c`: implementação em C das versões CPU do GEMM.
- `gemm_fallback.py`: script de apoio para gerar resultados quando a execução nativa não está disponível.
- `cpu_results.csv`: resultados obtidos dos experimentos de CPU.
- `gpu_results.csv`: resultados dos experimentos de GPU/estimados.
- `relatorio_gemm.txt`: relatório em texto do trabalho.
- `vec_report.txt`: relatório de vetorização gerado pelo compilador.

## Como usar

### 1. Abrir o notebook
Abra o arquivo `Trabalho Prático 3.ipynb` em Jupyter Notebook ou JupyterLab.

### 2. Executar os experimentos
O notebook já contém as células para:
- compilar e executar a implementação em C;
- gerar os resultados em CSV;
- plotar os gráficos;
- estruturar o relatório final.

### 3. Executar a versão de fallback em Python
Caso o ambiente não tenha o compilador C ou suporte completo para CUDA, você pode usar:

```bash
python gemm_fallback.py
```

## Observações importantes

- Os resultados de CPU foram obtidos a partir de execuções locais reais no ambiente disponível.
- A parte GPU foi preparada para execução com CUDA, mas a execução real depende de um ambiente com o CUDA Toolkit instalado e funcionando.

## Requisitos sugeridos

- Python 3
- NumPy
- pandas
- Jupyter Notebook ou JupyterLab
- compilador GCC para a parte em C
- CUDA Toolkit, se for executar a implementação GPU de forma real
