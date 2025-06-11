
## Delete vs code server:
rm -rf ~/.vscode-server
rm -rf ~/.vscode-remote

## Speicher checken:
df -h

## Leere Dateien finden:
find results/preprocessed_all -type f -empty

## Leere Dateien löschen:
find results/preprocessed_all -type f -empty -delete
