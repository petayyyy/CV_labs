# plot_grid_results.py
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import re

def plot_grid_stats(stats_file="frame_stats_grid.csv", output_plot="analysis_grid.png"):
    # Загрузка данных
    df = pd.read_csv(stats_file, delimiter=';')
    
    # Удаление нулевого кадра
    df = df[df['frame'] > 0].reset_index(drop=True)
    
    if df.empty:
        raise ValueError("После удаления кадра 0 данных не осталось.")
    
    # Определение количества областей
    detected_cols = [col for col in df.columns if re.match(r'detected_\d+', col)]
    region_ids = sorted([int(re.search(r'detected_(\d+)', col).group(1)) for col in detected_cols])
    
    if not region_ids:
        raise ValueError("Не найдено столбцов detected_N. Проверьте CSV.")
    
    print(f"Обнаружено {len(region_ids)} областей: {region_ids}")

    # Создание фигуры с сеткой 2x3
    plt.figure(figsize=(18, 10))

    # График 1: Общее время обработки кадра
    plt.subplot(2, 3, 1)
    plt.plot(df['frame'], df['total_ms_all'], label='Общее время', linewidth=1.5, color='black')
    plt.xlabel('Номер кадра')
    plt.ylabel('Время (мс)')
    plt.title('Общее время обработки')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # График 2: Обнаруженные признаки
    plt.subplot(2, 3, 2)
    for rid in region_ids:
        plt.plot(df['frame'], df[f'detected_{rid}'], label=f'Обл. {rid}', linewidth=1.5)
    plt.xlabel('Номер кадра')
    plt.ylabel('Количество')
    plt.title('Обнаруженные признаки')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # График 3: Сопоставленные признаки
    plt.subplot(2, 3, 3)
    for rid in region_ids:
        plt.plot(df['frame'], df[f'matched_{rid}'], label=f'Обл. {rid}', linewidth=1.5)
    plt.xlabel('Номер кадра')
    plt.ylabel('Количество')
    plt.title('Сопоставленные признаки')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # График 4: Время детекции по областям ← НОВЫЙ
    plt.subplot(2, 3, 4)
    for rid in region_ids:
        plt.plot(df['frame'], df[f'detect_ms_{rid}'], label=f'Обл. {rid}', linewidth=1.5)
    plt.xlabel('Номер кадра')
    plt.ylabel('Время (мс)')
    plt.title('Время детекции по областям')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # График 5: Время сопоставления по областям (опционально, но полезно)
    plt.subplot(2, 3, 5)
    for rid in region_ids:
        plt.plot(df['frame'], df[f'match_ms_{rid}'], label=f'Обл. {rid}', linewidth=1.5)
    plt.xlabel('Номер кадра')
    plt.ylabel('Время (мс)')
    plt.title('Время сопоставления по областям')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Сводная панель
    plt.subplot(2, 3, 6)
    plt.axis('off')
    total_detected = sum(df[f'detected_{rid}'] for rid in region_ids)
    total_matched = sum(df[f'matched_{rid}'] for rid in region_ids)
    total_detect_ms = sum(df[f'detect_ms_{rid}'] for rid in region_ids)
    total_match_ms = sum(df[f'match_ms_{rid}'] for rid in region_ids)

    stats_text = f"""
    Сводка (без кадра 0):

    Кадров: {len(df)}
    Областей: {len(region_ids)}

    Среднее (всего):
      Время:       {df['total_ms_all'].mean():.1f} мс
      Обнаружено:  {total_detected.mean():.1f}
      Сопоставлено:{total_matched.mean():.1f}

    Среднее (детекция):
      Общее:       {total_detect_ms.mean():.1f} мс
      На область:  {df[[f'detect_ms_{rid}' for rid in region_ids]].mean().mean():.2f} мс
    """
    plt.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center', family='monospace')

    plt.tight_layout()
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n✅ Графики по заданию 4 сохранены в: '{os.path.abspath(output_plot)}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Анализ статистики по областям (с детекцией и сопоставлением).')
    parser.add_argument('stats_file', nargs='?', default='frame_stats_grid.csv', help='CSV-файл со статистикой.')
    parser.add_argument('--output_plot', default='analysis_grid.png', help='Имя выходного файла.')
    args = parser.parse_args()

    plot_grid_stats(args.stats_file, args.output_plot)