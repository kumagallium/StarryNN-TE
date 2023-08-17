import matplotlib.pyplot as plt
import seaborn as sns

# Matplotlib settings
plt.rcParams["font.size"] = 11
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.major.width"] = 1.2
plt.rcParams["ytick.major.width"] = 1.2
plt.rcParams["xtick.major.size"] = 3
plt.rcParams["ytick.major.size"] = 3
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["axes.grid"] = False
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.linewidth"] = 0.3
plt.rcParams["legend.markerscale"] = 2
plt.rcParams["legend.fancybox"] = False
plt.rcParams["legend.framealpha"] = 1
plt.rcParams["legend.edgecolor"] = "black"


def parity_plot(target, true, pred):
    fig = plt.figure(figsize=(4, 4), dpi=300, facecolor="w", edgecolor="k")
    ax = fig.add_subplot(1, 1, 1)
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")

    ax.scatter(true, pred)

    if target == "Thermal conductivity":
        ax.set_xlabel("Experimental $\u03BA$ [Wm$^{-1}$K$^{-1}$]")
        ax.set_ylabel("Predicted  $\u03BA$ [Wm$^{-1}$K$^{-1}$]")
        ax_min = 0
        ax_max = 20
        ax.set_xlim(ax_min, ax_max)
        ax.set_ylim(ax_min, ax_max)
    elif target == "Seebeck coefficient":
        ax.set_xlabel("Experimental $S$ [\u03BCVK$^{-1}$]")
        ax.set_ylabel("Predicted $S$ [\u03BCVK$^{-1}$]")
        ax_min = 0
        ax_max = 800
        ax.set_xlim(ax_min, ax_max)
        ax.set_ylim(ax_min, ax_max)
    elif target == "Electrical conductivity":
        ax.set_xlabel("Experimental $\u03C3$ [\u03A9$^{-1}$m$^{-1}$]")
        ax.set_ylabel("Predicted  $\u03C3$ [\u03A9$^{-1}$m$^{-1}$]")
        ax_min = 0
        ax_max = 1000000
        ax.set_xlim(ax_min, ax_max)
        ax.set_ylim(ax_min, ax_max)
    elif target == "PF_calc":
        ax.set_xlabel("Experimental $PF_{ \mathrm{calc}}$ [mWm$^{-1}$K$^{-2}$]")
        ax.set_ylabel("Predicted $PF_{ \mathrm{calc}}$ [mWm$^{-1}$K$^{-2}$]")
        ax_min = 0
        ax_max = 10
        ax.set_xlim(ax_min, ax_max)
        ax.set_ylim(ax_min, ax_max)
    elif target == "ZT":
        ax.set_xlabel("Experimental $ZT$")
        ax.set_ylabel("Predicted $ZT$")
        ax_min = 0
        ax_max = 2
        ax.set_xlim(ax_min, ax_max)
        ax.set_ylim(ax_min, ax_max)

    ax.grid(True)
    ax.plot([ax_min, ax_max], [ax_min, ax_max], color="red")
    plt.tight_layout()
    plt.savefig("results/parity_plot_" + target.replace(" ", "_") + ".png")


def loss_plot(epochs, train_loss, val_loss):
    fig = plt.figure(figsize=(4, 4), dpi=300, facecolor="w", edgecolor="k")
    ax = fig.add_subplot(1, 1, 1)
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.set_xlim(min(epochs), max(epochs))
    ax.set_ylim(0, max(max(train_loss), max(val_loss)))

    ax.plot(epochs, train_loss, "bo", label="Training loss")
    ax.plot(epochs, val_loss, "b", label="Validation loss")
    ax.set_title("Training and Validation Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/loss_plot.png")


def pred_table(df_results, df_formula, prop):
    fig = plt.figure(figsize=(12, 12), dpi=400, facecolor="w", edgecolor="k")
    ax = fig.add_subplot(1, 1, 1)
    ax.tick_params(pad=1)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(bottom="off", top="off")
    ax.tick_params(left="off")
    ax.tick_params(bottom=False, left=False, right=False, top=False)
    rank = 100
    sns.heatmap(
        df_results.iloc[:rank],
        cmap="jet",
        annot=df_formula.iloc[:rank],
        fmt="",
        # vmin=0.5,
        # vmax=1,
        annot_kws={"size": 7},
        cbar_kws={"pad": 0.01},
    )
    plt.tight_layout()
    plt.savefig("results/pred_table_" + prop.replace(" ", "_") + ".png")
