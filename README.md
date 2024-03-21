# continual-pretrain
このリポジトリは、LLM（大規模言語モデル）を継続事前学習するために作成しました。  
環境構築は[dev-llmリポジトリ](https://github.com/oriki101/dev-llm)を参考にしてください。

## シングルノードでの学習
```bash
cd continual-pretrain
deepspeed src/train_deepspeed.py --train_config ./configs/train_configs/train_base.yaml
```

## マルチノードでの学習
国立研究開発法人産業技術総合研究所によって構築・運用されているABCI（AI Bridging Cloud Infrastructure）を利用してマルチノード学習を行います。DeepSpeedはデフォルトでPDSH（Parallel Distributed Shell）を使って分散学習を行いますが、ABCI環境ではSSH経由で接続したノード上でPythonが読み込めないことによりエラーが発生する場合があります。そのため、シングルノード学習のように**`deepspeed`**コマンドを用いるには、ソースコードの修正が必要です。しかし、この作業は環境構築の過程で大きな手間となります。

そこで、Open MPIの**`mpirun`**コマンドを使用して分散学習を行う方法を採用します。これにより、複雑な設定を避けつつ、効率的なマルチノード学習が可能になります。実行コマンドは以下になります。

```bash
cd continual-pretrain
sh script/continual_pretrain_abci.sh
```
