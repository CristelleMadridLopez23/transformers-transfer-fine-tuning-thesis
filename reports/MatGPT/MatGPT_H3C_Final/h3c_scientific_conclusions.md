# Conclusiones científicas H3-C

[HECHO VERIFICADO] La topología exacta es Ridge paralelo a MLP8 residual; no una cadena Ridge→MLP. Checkpoints reproducidos con error máximo 7.08e-07 eV/átomo. La etiqueta histórica es energía DFT por átomo.

[RESULTADO EXPERIMENTAL] H3-C preentrenado: MAE absoluto 183.3, residual 33.8 y de pares 58.8 meV/átomo. Acierto del estado fundamental elegible 53.2 % y pares >5 66.8 %. Composición sola consigue mejor MAE absoluto (52.0), pero produce empates en todos los tripletes.

[INFERENCIA] El backbone preentrenado aporta señal para ordenar polimorfos frente al aleatorio, con diferencia apareada de +12.52 [+5.53, +19.50] puntos porcentuales, pero el MAE de diferencias no mejora de forma concluyente (-3.38 [-8.72, +1.89] meV/átomo). No hay evidencia de que H3-C agote la información disponible en MatGPT ni de que añadir geometría resolverá el error.

[HIPÓTESIS] La geometría explícita podría añadir información a diferencias intrafórmula, especialmente si pares estructuralmente distintos tienen embeddings similares o gaps predichos comprimidos. Será una comparación posterior, con mismo objetivo y protocolo.

[LIMITACIÓN] Los 70 grupos de desarrollo ya influyeron en decisiones históricas; los intervalos son exploratorios. No se certificó una política XC/U/correcciones homogénea por registro gga-static. El test histórico permanece cerrado.
