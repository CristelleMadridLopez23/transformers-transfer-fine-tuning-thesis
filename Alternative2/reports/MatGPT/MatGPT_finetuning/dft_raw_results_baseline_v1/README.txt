Fuente: Alternative2/01_data/raw_data/results(1).json; 480 calculos DFT convergidos.
dft_polymorph_average.csv: medias y medianas por protocolo DFT (static/relax) y familia polimorfica.
dft_polymorph_by_composition.csv: una fila por calculo; conserva energia, fuerzas, stress, Delta-E y ranking dentro de composicion y estado magnetico.
Delta-E = E(polymorfo) - minimo DFT de la misma composicion y estado magnetico. Rank 1 usa la etiqueta de minimo DFT de la tabla fuente, incluyendo co-minimos por tolerancia; los Delta-E numericos se conservan sin redondear antes de exportar.
La energia absoluta por atomo se incluye como registro descriptivo, pero su promedio entre composiciones quimicamente distintas no es una referencia comparable; para ranking se usa Delta-E.
Force MAE/RMSE y stress MAE/RMSE son magnitudes de los residuos DFT respecto a cero, promediadas por estructura; no son errores de modelo.
Stress fue convertido de kbar a GPa. Hay 80 estructuras por familia y protocolo.
