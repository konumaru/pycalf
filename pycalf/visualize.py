import matplotlib.pyplot as plt


def plot_auuc(uplift_score, lift, baseline, auuc=None):
    label = f"AUUC = {auuc:.4f}" if auuc is not None else None

    plt.title('AUUC')
    plt.plot(lift, label=label)
    plt.plot(baseline)
    plt.xlabel("uplift score rank")
    plt.ylabel("lift")
    plt.legend(loc='lower right')
    plt.show()
