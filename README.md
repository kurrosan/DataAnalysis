Применение методов машинного обучения для обнаружения мошеннической активности в финансовых транзакциях
===========
[Данные, нужные для запуска кода](#title1)

[Запуск кода](#title2)

[Как работает код](#title3)

[Результат работы кода](#title4)

## <a id="title1">Данные, нужные для запуска кода</a>
Лабораторная_Технологии_Интел_Анализа_Данных.ipynb - основная программа   
Датасет [creditcard.csv](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)

## <a id="title2">Запуск кода</a>
Загрузите  [creditcard.csv](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data) в Google Colab  
![image](https://github.com/kurrosan/DataAnalysis/assets/120035199/bfe02eae-9b3b-4a03-ae5b-15d19a817bdf)

Или в Kaggle:  
![image](https://github.com/kurrosan/DataAnalysis/assets/120035199/5346a717-48c4-4725-8b34-bce180cf0d7f)

После этого запустите код 
После выполнения всего кода, у нас появится файл с названием submition.csv
![image](https://github.com/kurrosan/DataAnalysis/assets/120035199/81125b73-6254-4bcc-994a-57c5f900b21e)


## <a id="title3">Как работает код</a>
Этот код выполняет задачу анализа данных и построения модели классификации на основе Random Forest для определения мошеннических транзакций с кредитными картами. Загружаются необходимые библиотеки для работы с данными, построения модели и визуализации. Данные о транзакциях считываются из CSV-файла. Выделяются функции (признаки) и целевая переменная из данных. Данные разделяются на обучающий и тестовый наборы для оценки производительности модели. Создается и обучается модель Random Forest на обучающих данных. Модель применяется к тестовым данным для получения предсказаний. Рассчитываются метрики точности, матрица ошибок и отчет о классификации. Проводится поиск оптимальных параметров модели с использованием кросс-валидации. Выводятся лучшие параметры, найденные в результате GridSearchCV. Создается DataFrame с важностью признаков на основе обученной модели. Модель оценивается с использованием кросс-валидации для проверки ее обобщающей способности. Используется RandomOverSampler для балансировки классов в обучающем наборе. Важность признаков визуализируется с использованием столбчатой диаграммы. 


### В представленном коде есть несколько интересных и примечательных моментов:

1. Борьба с несбалансированными данными:
Используется библиотека imbalanced-learn, а именно RandomOverSampler, для увеличения количества экземпляров минорного класса (мошеннические транзакции) и балансировки классов.

2. Настройка параметров модели с помощью GridSearchCV:
Применяется метод кросс-валидации для поиска оптимальных параметров модели Random Forest. Это улучшает обобщающую способность модели и помогает избежать переобучения.

3. Визуализация важности признаков:
После настройки параметров модели строится столбчатая диаграмма важности признаков, что позволяет легко определить, какие функции оказывают наибольшее влияние на предсказания модели.

## <a id="title4">Результат работы кода</a>
Результатом работы является получение файла c предсказанием submition.csv
```python
y_pred = best_model.predict(X_test)

# Создание DataFrame с идентификаторами тестовых данных и предсказанными метками
submission_df = pd.DataFrame({'Id': range(1, len(X_test) + 1), 'PredictedLabel': y_pred})

# Запись DataFrame в CSV файл
submission_df.to_csv('submition.csv', index=False) 
```

