Применение методов машинного обучения для обнаружения мошеннической активности в финансовых транзакциях
===========
[Данные, нужные для запуска кода](#title1)

[Запуск кода](#title2)

[Как работает код](#title1)

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
Загружаем нужные библиотеки:
```python
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

Если вы загрузили данные на Google Диск, то можно воспользоваться данным кодом для импортирования файлов:
```python
from google.colab import drive
drive.mount('/content/gdrive')
```

Запись датасета в переменную data, сюда добавляется ссылка на датасет:
```python
data = pd.read_csv('/content/gdrive/MyDrive/Лабораторные работы/creditcard.csv')
```

Подготовка данных:
```python
X = data.drop('Class', axis=1)  # Функции (признаки)
y = data['Class']  # Целевая переменная
```

Разделение данных на обучающий и тестовый наборы
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Создание модели Random Forest и ее обучение
```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

Предсказание на тестовом наборе данных
```python
y_pred = model.predict(X_test)
```

Оценка качества модели
```python
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)#Здесь создается матрица ошибок
class_report = classification_report(y_test, y_pred)#Здесь создается отчет о классификации
```
Вывод результатов
```python
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')
```
```python
from sklearn.model_selection import GridSearchCV

# Задание сетки параметров для поиска
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Использование GridSearchCV для поиска оптимальных параметров
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Вывод лучших параметров
print("Best Parameters:", grid_search.best_params_)

# Использование лучших параметров для обучения модели
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)
```
Создание DataFrame с важностью признаков, где 'feature' - названия признаков, 'importance' - их важность, полученная из best_model.feature_importances_.
```python
feature_importances = pd.DataFrame({'feature': X.columns, 'importance': best_model.feature_importances_})
feature_importances = feature_importances.sort_values(by='importance', ascending=False)
print(feature_importances)
```
![image](https://github.com/kurrosan/DataAnalysis/assets/120035199/d1f7850b-9661-4cec-83f7-e43aa35a446e)

```python
from sklearn.model_selection import cross_val_score

# Кросс-валидация
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean CV Accuracy: {cv_scores.mean()}')
```
```python
from imblearn.over_sampling import RandomOverSampler

# Пример использования RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

# Обучение модели на увеличенном датасете
best_model.fit(X_resampled, y_resampled)
```
```python
import matplotlib.pyplot as plt
import numpy as np

# Вывод важности признаков
feature_importances = pd.DataFrame({'feature': X.columns, 'importance': best_model.feature_importances_})
feature_importances = feature_importances.sort_values(by='importance', ascending=False)

# Построение графика важности признаков
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importances)), feature_importances['importance'], align='center')
plt.xticks(range(len(feature_importances)), feature_importances['feature'], rotation=45)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()
```
![image](https://github.com/kurrosan/DataAnalysis/assets/120035199/282005f5-3fc1-4c45-8326-6a1b17323db7)

```python
y_pred = best_model.predict(X_test)

# Создание DataFrame с идентификаторами тестовых данных и предсказанными метками
submission_df = pd.DataFrame({'Id': range(1, len(X_test) + 1), 'PredictedLabel': y_pred})

# Запись DataFrame в CSV файл
submission_df.to_csv('submition.csv', index=False) 
```

