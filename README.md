# Fraud Transaction Monitor

Sistema experimental de detecção e monitoramento de fraudes em transações, com comparação de modelos, SQLite, alertas explicáveis, API REST e dashboard interativo para VS Code.

## Resultados da versão 2

Avaliação final cronológica, sem usar o teste para escolher o limite:

| Métrica | Resultado |
|---|---:|
| Modelo selecionado | Regressão logística balanceada |
| Precisão | 42,2% |
| Recall | 82,7% |
| F1 | 55,9% |
| PR-AUC | 76,4% |
| Alertas no conjunto completo | 823 |

A versão anterior tinha aproximadamente 5% de precisão e gerava 6.680 alertas. Os resultados são experimentais e não autorizam bloqueio automático de pagamentos.

## Arquitetura

```text
fraud-transaction-monitor/
├── src/fraud_monitor/
│   ├── constants.py       # variáveis e níveis de risco
│   ├── database.py        # importação e esquema SQLite
│   ├── metrics.py         # PR-AUC, precisão, recall e threshold
│   ├── models.py          # Gaussian NB e regressão logística
│   └── pipeline.py        # treino, seleção e scoring
├── tests/                 # testes automatizados
├── work/
│   ├── build_dashboard.py
│   └── dataset/           # ignorado pelo Git
├── outputs/               # banco, modelo e dashboard
├── api_server.py
├── fraud_detector.py
└── .github/workflows/ci.yml
```

## Executar no VS Code

Abra **Terminal → Run Task** e execute na ordem:

1. `1. Instalar dependências`
2. `2. Importar CSV`
3. `3. Treinar modelos`
4. `4. Gerar alertas`
5. `5. Atualizar painel`
6. `6. Servir dashboard no VS Code`

Para abrir o painel dentro do editor, pressione `Ctrl+Shift+P`, escolha **Simple Browser: Show** e informe:

```text
http://localhost:8000/painel-transacoes-suspeitas.html
```

O painel oferece métricas, distribuição por risco, busca, filtros, paginação, justificativas e exportação CSV.

## Dataset

Baixe `creditcard.csv` no [dataset Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) e coloque em `work/dataset/`.

O dataset, ZIP e banco SQLite são ignorados pelo Git. As variáveis `V1`–`V28` são componentes PCA anonimizados; por isso, a explicação indica contribuição estatística, não motivos comerciais como localização ou dispositivo.

## API REST

Execute a tarefa `8. Iniciar API`. Endereço padrão: `http://localhost:8080`.

| Método | Rota | Uso |
|---|---|---|
| GET | `/health` | Saúde e versão do modelo |
| GET | `/alerts?risk=critical&limit=100` | Listar alertas |
| GET | `/alerts/{transaction_id}` | Detalhar um alerta |
| PATCH | `/alerts/{transaction_id}` | Atualizar revisão |
| POST | `/transactions/analyze` | Analisar nova transação |

Exemplo de atualização:

```json
{"review_status": "investigating"}
```

O endpoint de análise exige `Time`, `Amount` e `V1`–`V28`, pois o modelo foi treinado nessas variáveis.

## Avaliação correta

Os dados são ordenados por tempo e divididos em:

- 60% treino;
- 20% validação, usada para seleção do modelo e threshold;
- 20% teste, utilizado apenas na avaliação final.

O pipeline compara Gaussian Naive Bayes e regressão logística balanceada pela PR-AUC da validação. O threshold busca pelo menos 80% de recall e maximiza a precisão.

## Testes e CI

Execute a tarefa `9. Executar testes` ou:

```powershell
python -m unittest discover -s tests -v
```

O GitHub Actions valida os testes e a sintaxe Python em cada push e pull request.

## Limitações

- Dataset histórico, anonimizado e restrito a dois dias de 2013.
- Probabilidades não devem ser tratadas como garantia de fraude.
- Níveis de risco são relativos ao threshold calibrado.
- Produção exige variáveis reais, monitoramento de drift, auditoria, segurança e revisão humana.
