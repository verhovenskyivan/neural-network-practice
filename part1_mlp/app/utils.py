"""Дополнительные функции отображения графика обучения."""


def save_loss_plot(history, filename):
    """Сохраняет PNG-график изменения бинарной кросс-энтропии."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
    except ImportError as error:
        raise RuntimeError('Для графика установите matplotlib.') from error
    if not history:
        raise ValueError('История обучения пуста.')
    fig, axes = plt.subplots(figsize=(7, 3.5))
    axes.plot(range(1, len(history) + 1), history)
    axes.set(title='Динамика ошибки обучения', xlabel='Эпоха', ylabel='Binary cross-entropy')
    axes.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
