# plot_results.py
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_stats(stats_file="frame_stats.csv", output_plot="analysis.png"):
    # Загрузка данных
    df = pd.read_csv(stats_file, delimiter=';')

    # Создание графиков
    plt.figure(figsize=(16, 9))

    # График 1: Время обработки (1, 2, 3)
    plt.subplot(2, 2, 1)
    plt.plot(df['frame'], df['total_ms'], label='Общее время', linewidth=1.5)
    plt.plot(df['frame'], df['detect_ms'], label='Обнаружение', linewidth=1.5)
    plt.plot(df['frame'], df['match_ms'], label='Сопоставление', linewidth=1.5)
    plt.xlabel('Номер кадра')
    plt.ylabel('Время (мс)')
    plt.title('Время обработки кадра')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # График 2: Время на признак (4)
    plt.subplot(2, 2, 2)
    # Избегаем деления на ноль
    df['time_per_feature'] = df['total_ms'] / df['detected'].clip(lower=1)
    plt.plot(df['frame'], df['time_per_feature'], color='purple', linewidth=1.5)
    plt.xlabel('Номер кадра')
    plt.ylabel('Время на признак (мс)')
    plt.title('Время обработки на один признак')
    plt.grid(True, linestyle='--', alpha=0.7)

    # График 3: Количество признаков (5, 6)
    plt.subplot(2, 2, 3)
    plt.plot(df['frame'], df['detected'], label='Обнаружено', linewidth=1.5)
    plt.plot(df['frame'], df['matched'], label='Сопоставлено', linewidth=1.5)
    plt.xlabel('Номер кадра')
    plt.ylabel('Количество признаков')
    plt.title('Количество признаков')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Информационная панель
    plt.subplot(2, 2, 4)
    plt.axis('off')
    stats_text = f"""
    Сводка по обработке:
    
    Всего кадров: {len(df)}
    Среднее время обработки: {df['total_ms'].mean():.1f} мс
    Среднее количество признаков: {df['detected'].mean():.1f}
    Среднее количество совпадений: {df['matched'].mean():.1f}
    """
    plt.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')

    plt.tight_layout()
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nГрафики построены и сохранены в '{os.path.abspath(output_plot)}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot statistics from feature tracking.')
    parser.add_argument('stats_file', nargs='?', default='frame_stats.csv', help='Файл со статистикой.')
    parser.add_argument('--output_plot', default='analysis.png', help='Имя файла для графиков.')
    args = parser.parse_args()

    plot_stats(args.stats_file, args.output_plot)