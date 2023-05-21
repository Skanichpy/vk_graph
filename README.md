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
### <u>Запуск</u>:
Последовательный запуск .ipynb Stage1->...Stage4!

<!-- ![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here) -->
<br>

[![presentation](https://img.shields.io/badge/Microsoft_PowerPoint-B7472A?style=for-the-badge&logo=microsoft-powerpoint&logoColor=white&label=%D0%9F%D1%80%D0%B5%D0%B7%D0%B5%D0%BD%D1%82%D0%B0%D1%86%D0%B8%D1%8F)](https://disk.yandex.ru/i/BBSeICT6dVIuLg)
