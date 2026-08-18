# Fraud Transaction Monitor

Sistema experimental de detecção e monitoramento de fraudes em transações, com modelo probabilístico, SQLite, alertas explicáveis e dashboard interativo para VS Code.

O projeto importa o dataset ULB/Kaggle, treina um classificador, calcula o risco de cada transação, registra alertas e apresenta justificativas técnicas em um painel pesquisável.

## Estrutura

```text
.
├── .vscode/                       # tarefas, depuração e extensões
├── outputs/
│   ├── fraud_model.json           # modelo e métricas
│   └── painel-transacoes-suspeitas.html
├── work/
│   ├── build_dashboard.py         # gerador do painel
│   └── dataset/creditcard.csv     # local; não enviado ao GitHub
├── fraud_detector.py
├── requirements.txt
└── README.md
```

O banco `outputs/fraud_detection.db`, o CSV e o ZIP do Kaggle são ignorados pelo Git devido ao tamanho.

## Passo a passo no VS Code

### 1. Abrir o projeto

No VS Code, escolha **File → Open Folder** e abra a pasta deste projeto. Como alternativa, execute `code .` nesta pasta.

### 2. Instalar as extensões

Aceite a recomendação do VS Code para instalar Python, Pylance e SQLite Viewer.

### 3. Conferir o dataset

O arquivo precisa existir em `work/dataset/creditcard.csv`. Se estiver começando em outro computador, baixe-o no [dataset do Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

### 4. Executar as tarefas

Abra **Terminal → Run Task** e execute, aguardando cada uma terminar:

1. `1. Instalar dependências`
2. `2. Importar CSV`
3. `3. Treinar modelo`
4. `4. Gerar alertas`
5. `5. Atualizar painel`
6. `6. Servir dashboard no VS Code`

Na primeira execução, faça todas as etapas. Para apenas atualizar a visualização após novos alertas, execute as tarefas 5 e 6.

### Abrir o dashboard dentro do VS Code

1. Execute **Terminal → Run Task → 6. Servir dashboard no VS Code** e deixe esse terminal aberto.
2. Pressione `Ctrl+Shift+P`.
3. Procure e selecione **Simple Browser: Show**.
4. Informe `http://localhost:8000/painel-transacoes-suspeitas.html`.
5. O dashboard abrirá em uma aba interna do VS Code.

Se preferir o navegador normal, execute a tarefa `7. Abrir painel externo`.

O dashboard lista o banco, o modelo e o próprio painel, além dos filtros de transações suspeitas.

### Entender a justificativa

`Fraude confirmada` significa que a transação possui `Class = 1` no dataset original; não é uma confirmação produzida pelo modelo. A coluna **Justificativa** apresenta os três atributos que mais aproximaram a transação do padrão estatístico de fraude. Valores positivos entre parênteses indicam força favorável à classe fraudulenta. Como `V1` a `V28` são componentes PCA anonimizados, eles não permitem justificativas comerciais como localização, dispositivo ou estabelecimento.

### 5. Examinar o banco

No explorador do VS Code, abra `outputs/fraud_detection.db` com SQLite Viewer. Consulta útil:

```sql
SELECT a.transaction_id, a.fraud_probability, t.Amount, t.Time, a.actual_class
FROM alerts AS a
JOIN transactions AS t ON t.rowid = a.transaction_id
ORDER BY a.fraud_probability DESC;
```

### 6. Depurar o código

Abra **Run and Debug**, pressione `F5` e escolha `Treinar detector` ou `Gerar alertas`. Você pode adicionar pontos de interrupção no arquivo `fraud_detector.py`.

## Publicar no GitHub

Crie um repositório vazio e execute:

```powershell
git init
git add .
git commit -m "Cria detector de transações suspeitas"
git branch -M main
git remote add origin https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git
git push -u origin main
```

Revise `git status` antes do commit. O `.gitignore` evita enviar o banco, o dataset e ambientes locais.

## Limitações

Este projeto é uma demonstração. As variáveis `V1` a `V28` são anonimizadas, e o modelo não deve bloquear pagamentos reais sem validação, calibração e revisão humana.

