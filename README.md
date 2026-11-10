# Detección de imágenes generadas por inteligencia artificial mediante clasificación binaria

Código y resultados del trabajo final de la Maestría en Ciencias de los Datos y la Analítica (Universidad EAFIT, 2026),
de Isabel Jurado Jaramillo, con la asesoría de Jorge Iván Padilla. El trabajo compara EfficientNet-B4 y ViT-B/16
en la clasificación de imágenes reales frente a imágenes generadas por IA con el
[Defactify Image Dataset](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset).

## Contenido

| Archivo | Para qué sirve |
|---|---|
| `EDA_ampliado_v2.ipynb` | Análisis exploratorio: color e intensidad, micro-textura, espectro, ruido, formato y fuga por escenas. |
| `comparacion_efficientnet_vit4.ipynb` | Notebook principal: partición por escena, entrenamiento de los dos modelos, validación cruzada de 5 pliegues, métricas, intervalos bootstrap, pruebas estadísticas y exportación del modelo. |
| `inferencia_real_vs_ia.ipynb` | Uso del modelo exportado para clasificar imágenes nuevas. |
| `curvas_efficientnet.csv`, `curvas_vit.csv` | Métricas de validación durante el entrenamiento de cada modelo. |
| `predicciones_val.csv`, `predicciones_test.csv` | Probabilidades de los dos modelos en validación y prueba. |
| `cv_efficientnet_fold1.json` … `cv_vit_fold5.json` | Resultados de cada pliegue de la validación cruzada. |

## Pesos del modelo

El modelo exportado (`best_model_real_vs_ia_v4.pt`, ViT-B/16, 327 MB) supera el límite de tamaño de GitHub y se
descarga desde: **[agregar aquí el enlace de Hugging Face, Zenodo o Drive]**.

El archivo contiene los pesos, la arquitectura (`vit_base_patch16_224`), el preprocesamiento (lado corto a 224 px y
recorte central), la normalización (0.5 por canal), el umbral elegido en validación (≈0.0024) y el mapeo de
etiquetas (0 = real, 1 = generada).

## Cómo reproducir

1. Abrir los notebooks en Google Colab con GPU. Cada uno instala sus dependencias en la primera celda
   (PyTorch, Lightning, timm, datasets, TorchMetrics, scikit-learn, statsmodels, pandas, matplotlib y seaborn).
2. El dataset se descarga solo desde Hugging Face con `load_dataset("Rajarshi-Roy-research/Defactify_Image_Dataset")`.
3. Ejecutar `EDA_ampliado_v2.ipynb` (independiente de los demás).
4. Ejecutar `comparacion_efficientnet_vit4.ipynb` completo. Guarda resultados intermedios, así que si la sesión se
   interrumpe, al volver a correrlo no repite los entrenamientos ya terminados. La semilla es 42.
5. Usar `inferencia_real_vs_ia.ipynb` con el archivo de pesos para clasificar imágenes nuevas (funciona en CPU).

Los resultados se obtuvieron con una GPU NVIDIA RTX PRO 6000 Blackwell en Google Colab. Las versiones de las
librerías no se fijaron, por lo que con otras versiones o con otra GPU los números pueden variar ligeramente.

## Advertencia

El modelo se entrenó y evaluó con los cinco generadores del Defactify Image Dataset; no se probó con generadores
externos. El umbral guardado se eligió con imágenes sin alterar y conviene recalibrarlo para imágenes recomprimidas
o reducidas, como las que circulan en redes sociales. Úsese como apoyo a la revisión humana, no como veredicto.
