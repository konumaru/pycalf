# Documantation

## Build Docs
```
$ poetry run sphinx-apidoc -f -o ./docs .
ファイル ./docs/pycalf.rst を作成しています。
ファイル ./docs/tests.rst を作成しています。
ファイル ./docs/modules.rst を作成しています。

$ poetry run sphinx-build -b singlehtml ./docs ./docs/_build
Sphinx v3.2.1 を実行中
保存された環境データを読み込み中... 完了
ビルド中 [mo]: 更新された 0 件のpoファイル
ビルド中 [singlehtml]: all documents
環境データを更新中0 件追加, 0 件更新, 0 件削除
更新されたファイルを探しています... 見つかりませんでした
preparing documents... 完了
ドキュメントを1ページにまとめています... pycalf 完了
書き込み中... 完了
追加のファイルを出力... 完了
静的ファイルをコピー中... ... 完了
extraファイルをコピー中... 完了
オブジェクト インベントリを出力... 完了
ビルド 成功.

HTML ページはdocs/_buildにあります。
```
