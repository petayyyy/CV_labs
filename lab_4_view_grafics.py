# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_and_validate_csv(filename):
    try:
        df = pd.read_csv(filename, sep=';', skipinitialspace=True)
        return df
    except Exception as e:
        print(f"Ошибка при загрузке файла {filename}: {str(e)}")
        return None

# Загрузка данных с проверкой
df1 = load_and_validate_csv('points.csv')
df2 = load_and_validate_csv('pointsEq.csv')

if df1 is not None and df2 is not None:
    # Создание фигуры с двумя подграфиками
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Основной график сравнения
    ax1.plot(df1['Frame'], df1['Count points'], 'g-', label='Без эквализации')
    ax1.plot(df2['Equalized Frame'], df2['Count points'], 'r-', label='С эквализацией')
    ax1.set_title('Сравнение количества обнаруженных признаков')
    ax1.set_ylabel('Количество точек')
    ax1.legend()
    ax1.grid(True)
    
    # График разницы
    diff = df2['Count points'] - df1['Count points']
    ax2.bar(df1['Frame'], diff, color=np.where(diff >= 0, 'r', 'g'))
    ax2.set_xlabel('Номер кадра')
    ax2.set_ylabel('Разница (с экв. - без)')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('detailed_comparison.png', dpi=300)
    plt.show()
else:
    print("Не удалось загрузить данные для построения графиков")