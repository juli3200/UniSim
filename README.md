# UniSim

## Beschreibung

UniSim ist das Produkt einer Maturitätsarbeit an der Kantonsschule Uster. 
Die Arbeit trägt den Titel:
„Evolution und kooperatives Verhalten von Einzellern – eine computerbasierte Simulation“
und wurde von Julian Heer verfasst.

Das Programm simuliert Einzeller und versucht evolutionäre Mechanismen zu simulieren. 


## Experimente
Alle Rohdaten für die in der Arbeit besprochenen Experimente befinden sich im
[experiments](https://github.com/juli3200/UniSim/tree/main/experiments) Ordner.

## Anleitung

Laden Sie den neusten verfügbaren [release](https://github.com/juli3200/UniSim/releases/tag/v1.0.0) herunter. Falls eine CUDA kompatible NVIDIA Grafikkarte verfügbar ist, können sie die ```UniSim_gpu``` Version laden. 
Falls nicht, verwenden Sie die ```UniSim``` Version. Vor dem starten des Programms muss eine ```config.json``` Datei angelegt werden. In Ihr können Simulationsparameter (vgl. Abschnitt 3.2.4)
angepasst werden. Falls kein Pfad angegeben wird, werden die Standardwerte verwendet.

Mit ```help``` können alle verfügbaren Commands abgebildet werden. Um die Welt zu speichern muss der ```save``` Command __vor__ dem Start der  Simulation aufgerufen werden.

Um die Simulation anzuschauen, verwenden Sie das ```view``` Programm. Für andere Analysen verwenden Sie die Python Skripte im ```Scripts``` Ordner.

### Fortgeschrittene Installation
Um kompliziertere Programme mit der ```UniSim``` Bibliothek zu schreiben, muss dirketer Zugriff darauf genommen werden. Dafür werden [Rust](https://rust-lang.org/tools/install/) und [CUDA C++](https://developer.nvidia.com/cuda-downloads) (Optional) benötigt.
Falls Sie keine kompatible NVIDIA Grafikkarte besitzen, muss das feature ```cuda``` im ```Cargo.toml``` deaktiviert werden. 

Führen sie folgende Commands aus falls Sie das Cuda feature aktiviert haben:
```
nvcc -lib  src/cuda/cu_src/grid.cu -o .\native\windows\grid.lib
nvcc -lib  src/cuda/cu_src/memory.cu -o .\native\windows\memory.lib
nvcc -lib  src/cuda/cu_src/test.cu -o .\native\windows\test.lib
```

Sie können nun auf die Bibliothek zugreifen.





