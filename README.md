# vk_graph

Данный репозиторий содержит решение кейса от VK по прогнозированию интенсивности взаимодействия пользователей.
### <u>Структура</u>: 
```
├── data 
│   ├── attr.csv
│   ├── train.csv 
│   ├── test.csv
├── training 
│   ├── __init__.py
│   ├── trainer.py
├── Step1_get_graph_clusters.ipynb
├── Step2_graph_get_features.ipynb
├── Step3_preproc_x1_features.ipynb
├── Step4_fit_eval.ipynb
└── .gitignore
```
### <u>Описание</u>:
```
├── data - данные необходимые для запуска (атрибуты вершин-пользователей эго-графов)
├── training - модуль для подбора гипермапарметров моделей 
├── Step1_get_graph_clusters.ipynb - запуск кластеризации эго-графов 
├── Step2_graph get features.ipynb - генерация признаков на основе интенсивности 
├── Step3_preproc_x1_features.ipynb - предобработка сгенерированных признаков на прошлом этапе, а также разделение на кластеры 
├── Step4_fit_eval.ipynb - обучение и оценка построенных моделей для каждого кластера
```
