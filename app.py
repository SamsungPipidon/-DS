import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Загружаем модель и трансформер
loaded_model = pickle.load(open("/Users/egor24494icloud.com/Downloads/VKR-main/Models/model_regressor.sav", 'rb'))
min_max_scaler = pickle.load(open("/Users/egor24494icloud.com/Downloads/VKR-main/Models/model_transformer.sav", 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Получение данных из формы
        try:
            matrix = 0
            density = float(request.form.get('density', 0))
            elasticity = float(request.form.get('elasticity', 0))
            hardener = float(request.form.get('hardener', 0))
            epoxy = float(request.form.get('epoxy', 0))
            temperature = float(request.form.get('temperature', 0))
            surface_density = float(request.form.get('surface_density', 0))
            elastic_modulus = float(request.form.get('elastic_modulus', 0))
            toughness = float(request.form.get('toughness', 0))
            resin_consumption = float(request.form.get('resin_consumption', 0))
            patch_angle = float(request.form.get('patch_angle', 0))
            patch_pitch = float(request.form.get('patch_pitch', 0))
            patch_density = float(request.form.get('patch_density', 0))

            # Формирование списка входных данных
            input_data = [matrix, density, elasticity, hardener, epoxy, temperature, 
                          surface_density, elastic_modulus, toughness, resin_consumption, 
                          patch_angle, patch_pitch, patch_density]

            # Прогнозирование
            prediction = y_prediction_with_normalization(input_data)

            return render_template('index.html', prediction=f'Модуль упругости при растяжении, ГПа = {prediction}')
        except Exception as e:
            return render_template('index.html', error=f'Ошибка: {e}')
    else:
        return render_template('index.html', prediction=None, error=None)

def y_prediction_with_normalization(input_data):
    cols = ['Соотношение матрица-наполнитель', 'Плотность, кг/м3',
           'модуль упругости, ГПа', 'Количество отвердителя, м.%',
           'Содержание эпоксидных групп,%_2', 'Температура вспышки, С_2',
           'Поверхностная плотность, г/м2', 
           'Модуль упругости при растяжении, ГПа',
           'Прочность при растяжении, МПа', 'Потребление смолы, г/м2',
           'Угол нашивки, град', 'Шаг нашивки', 'Плотность нашивки']
    input_359 = np.array(input_data).reshape(1, -1)
    df = pd.DataFrame(input_359, columns=cols)
    
    df_w = pd.DataFrame(min_max_scaler.transform(df), columns=cols)
    
    y_pred = loaded_model.predict(df_w.drop('Модуль упругости при растяжении, ГПа', axis=1))
    
    df_w['Модуль упругости при растяжении, ГПа'] = y_pred
    new_df = pd.DataFrame(min_max_scaler.inverse_transform(df_w), columns=cols)
    return new_df['Модуль упругости при растяжении, ГПа'][0]

if __name__ == '__main__':
    app.run(debug=True)
