### Visual-based relatives finder for Genotek Family Tree Matcher project

`FIW_embedding_generation.ipynb` - генерация эмбедиингов для диц с фотографий из датасета FIW

`haystack_test.json` - пример данных о семье в формате от Генотека

`FIW_to_json.ipynb` - перевод таблиц родства из FIW в этот формат

`family_from_json.ipynb` - загрузка семей из этого формата в объекты Python

`FIW_training.ipynb` - обучение модели классификатора на FIW

`test_inference.ipynb` - оценка качества алгоритма с помощью симуляций на датасете FIW

`preparing_for_inference.ipynb` - подготовка json-файла для примера полного инференса

`inference_example.ipynb` - пример процесса инференса (может получать на вход json-файл с информацией о семьях и путями к фотографиям, выдает рекомендации возможных родственников для семей)