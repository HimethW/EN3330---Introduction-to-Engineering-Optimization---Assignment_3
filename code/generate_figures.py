# EN3330 Assignment 03 - builds the five figures used in the report.

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

f = lambda x: x**2
h = lambda x: 1 - x
L = lambda mu, x: f(x) + mu * h(x)
d = lambda mu: mu - mu**2 / 4

x_star, mu_star, p_star = 1.0, 2.0, 1.0

# mu and x are ordered sweeps, so the series use a light-to-dark blue ramp
# rather than unrelated hues. Line styles vary too, for greyscale printing.
RAMP4 = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]
RAMP3 = ["#86b6ef", "#3987e5", "#184f95"]
STYLES = ["-", "--", "-.", ":"]
ACCENT = "#eb6834"
INK, INK2 = "#0b0b0b", "#52514e"
INFEAS = "#f0efec"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.grid": True,
    "grid.color": "#d9d8d4",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)


def tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("wrote", name)


def fig_q1a():
    """L(mu,x) against x for mu = 0,1,2,3, with the minimisers marked."""
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    x = np.linspace(-1, 3, 800)

    ax.axvspan(-1, 1, color=INFEAS, zorder=0)
    ax.text(-0.92, 8.3, "infeasible\n$x<1$", color=INK2, fontsize=9, va="top")
    ax.axvline(x_star, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=1)

    for i, mu in enumerate([0, 1, 2, 3]):
        ax.plot(x, L(mu, x), color=RAMP4[i], ls=STYLES[i], lw=2.0, zorder=3)
        xm = mu / 2
        ax.plot([xm], [L(mu, xm)], "o", ms=8, color=RAMP4[i],
                mec="white", mew=1.5, zorder=5)
        ax.text(3.03, L(mu, 3.0), rf"$\mu={mu}$", color=RAMP4[i],
                fontsize=10, va="center", ha="left")

    mus = np.linspace(0, 3, 200)
    ax.plot(mus / 2, d(mus), color=ACCENT, lw=1.6, ls=(0, (2, 2)), zorder=4)
    ax.annotate(r"$x^\star(\mu)=\mu/2$", xy=(1.35, d(2.7)), xytext=(1.60, 0.10),
                color=ACCENT, fontsize=9, ha="left",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.0))
    ax.annotate(r"$\mu=2$: the minimiser" "\n" r"reaches the boundary",
                xy=(1.0, 1.06), xytext=(0.02, 4.25), color=INK,
                fontsize=9, ha="left",
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.0))

    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$L(\mu,x)=x^{2}+\mu(1-x)$")
    ax.set_title(r"Q1(a)  Lagrangian against $x$ for increasing $\mu$", pad=10)
    ax.set_xlim(-1, 3.45)
    ax.set_ylim(-0.3, 9.0)
    tidy(ax)
    save(fig, "q1a_lagrangian.pdf")


def fig_q1b():
    """The dual function, its maximiser, and the duality gap."""
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    mu = np.linspace(0, 5, 600)

    ax.fill_between(mu, d(mu), p_star, where=(d(mu) <= p_star),
                    color=INFEAS, zorder=0)
    ax.axhline(p_star, color=ACCENT, lw=1.6, ls=(0, (4, 3)), zorder=2)
    ax.text(4.25, p_star + 0.12, r"$p^\star=1$", color=ACCENT, fontsize=10)
    ax.text(3.60, 0.20, "duality gap\n$p^\\star-d(\\mu)$", color=INK2,
            fontsize=9, ha="center")

    ax.plot(mu, d(mu), color=RAMP4[2], lw=2.2, zorder=3)
    ax.text(0.55, d(0.55) - 0.42, r"$d(\mu)=\mu-\mu^{2}/4$",
            color=RAMP4[3], fontsize=10)

    ax.plot([mu_star], [d(mu_star)], "o", ms=9, color=RAMP4[3],
            mec="white", mew=1.5, zorder=5)
    ax.annotate(r"$\mu^\star=2,\;\; d^\star=1$", xy=(mu_star, d(mu_star)),
                xytext=(2.45, 1.52), color=INK, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.0))
    ax.vlines(mu_star, -1.4, d(mu_star), color=INK2, lw=0.9, ls=(0, (2, 3)))
    for m in (0, 1, 2, 3):
        ax.plot([m], [d(m)], "o", ms=5, color=RAMP4[3], zorder=4)

    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$d(\mu)=\min_x L(\mu,x)$")
    ax.set_title(r"Q1(b)  Dual function: concave, maximised at $\mu^\star=2$"
                 "\n" r"where $d^\star=p^\star$ (zero duality gap)", pad=10)
    ax.set_xlim(0, 5)
    ax.set_ylim(-1.4, 1.95)
    tidy(ax)
    save(fig, "q1b_dual.pdf")


def fig_q1c():
    """g(x) = max over mu, shown as the limit of partial maxima."""
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    x = np.linspace(-1, 3, 800)
    top = 9.1

    ax.axvspan(-1, 1, color=INFEAS, zorder=0)
    ax.axvline(x_star, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=1)

    for i, M in enumerate([2, 5, 15]):
        partial = np.where(x >= 1, f(x), L(M, x))
        ax.plot(x, partial, color=RAMP3[i], ls=STYLES[i + 1], lw=1.7, zorder=2)
        vis = np.flatnonzero(partial <= top)        # label the first on-screen point
        ax.text(x[vis[0]] + 0.06, min(partial[vis[0]], top) - 0.15,
                rf"$M={M}$", color=RAMP3[i], fontsize=9, ha="left", va="top")

    ax.plot(x[x >= 1], f(x[x >= 1]), color=INK, lw=2.6, zorder=4)
    ax.annotate("", xy=(1.0, 9.6), xytext=(1.0, 1.0),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.6))
    ax.text(0.93, 6.3, r"$g(x)=+\infty$", color=INK, fontsize=11,
            rotation=90, ha="right", va="center")
    ax.text(2.08, 3.6, r"$g(x)=x^{2}$", color=INK, fontsize=11)
    ax.text(-0.95, 0.15, "infeasible  $x<1$", color=INK2, fontsize=9,
            va="bottom")

    ax.plot([x_star], [p_star], "o", ms=9, color=ACCENT,
            mec="white", mew=1.5, zorder=6)
    ax.annotate(r"$\min_x g(x)=1$ at $x=1$", xy=(1.0, 1.0), xytext=(1.42, 0.55),
                color=ACCENT, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.0))

    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$g(x)=\max_{\mu\geq 0} L(\mu,x)$")
    ax.set_title(r"Q1(c)  Primal function: the inner maximisation builds an"
                 "\n" r"infinite wall on the infeasible side", pad=10)
    ax.set_xlim(-1, 3)
    ax.set_ylim(-0.4, 10.0)
    tidy(ax)
    save(fig, "q1c_primal.pdf")


def fig_q2a():
    """The two slices through the saddle point."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.6, 3.4))

    mu = np.linspace(0, 5, 400)
    axL.plot(mu, L(mu, x_star), color=RAMP4[2], lw=2.2, zorder=3)
    axL.plot([mu_star], [L(mu_star, x_star)], "o", ms=9, color=ACCENT,
             mec="white", mew=1.5, zorder=5)
    axL.axhline(1.0, color=ACCENT, lw=1.0, ls=(0, (2, 3)), zorder=1)
    axL.annotate(r"$L(\mu^\star,x^\star)=1$", xy=(mu_star, 1.0),
                 xytext=(2.45, 0.48), color=INK, fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=INK2, lw=1.0))
    axL.text(0.2, 1.62, r"$L(\mu,x^\star)=1+\mu\cdot 0=1$", color=RAMP4[3],
             fontsize=9)
    axL.set_xlabel(r"$\mu$")
    axL.set_ylabel(r"$L(\mu,\,x^\star)$")
    axL.set_title(r"flat in $\mu$: left inequality" "\n" r"holds with equality",
                  fontsize=10, pad=8)
    axL.set_xlim(0, 5)
    axL.set_ylim(0, 2.1)
    tidy(axL)

    x = np.linspace(-0.6, 2.6, 500)
    axR.plot(x, L(mu_star, x), color=RAMP4[2], lw=2.2, zorder=3)
    axR.plot([x_star], [L(mu_star, x_star)], "o", ms=9, color=ACCENT,
             mec="white", mew=1.5, zorder=5)
    axR.axhline(1.0, color=ACCENT, lw=1.0, ls=(0, (2, 3)), zorder=1)
    axR.annotate(r"$L(\mu^\star,x^\star)=1$", xy=(x_star, 1.0),
                 xytext=(1.25, 2.1), color=INK, fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=INK2, lw=1.0))
    axR.text(0.05, 3.05, r"$L(\mu^\star,x)=(x-1)^{2}+1$", color=RAMP4[3],
             fontsize=9)
    axR.set_xlabel("$x$")
    axR.set_ylabel(r"$L(\mu^\star,\,x)$")
    axR.set_title(r"minimised at $x^\star=1$: right" "\n" r"inequality holds",
                  fontsize=10, pad=8)
    axR.set_xlim(-0.6, 2.6)
    axR.set_ylim(0, 3.4)
    tidy(axR)

    fig.suptitle(r"Q2(a)(iv)  Two slices through the saddle point "
                 r"$(x^\star,\mu^\star)=(1,2)$; both pass through the value $1$",
                 fontsize=11, y=1.04)
    save(fig, "q2a_saddle.pdf")


def fig_q3b():
    """L as straight lines in mu for x = 0,1,2, and their lower envelope."""
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    mu = np.linspace(0, 5, 600)

    lines = []
    for i, x in enumerate([0, 1, 2]):
        y = L(mu, x)                            # affine in mu
        lines.append(y)
        ax.plot(mu, y, color=RAMP3[i], ls=STYLES[i], lw=1.8, zorder=2)
        ax.text(5.05, L(5.0, x), rf"$x={x}$", color=RAMP3[i], fontsize=10,
                va="center", ha="left")

    ax.plot(mu, np.min(lines, axis=0), color=ACCENT, lw=2.8, zorder=4)
    ax.text(1.35, 1.18, "lower envelope\n(concave)", color=ACCENT, fontsize=10,
            ha="center")

    ax.plot(mu, d(mu), color=INK, lw=1.6, ls=(0, (5, 2)), zorder=3)
    ax.annotate(r"true $d(\mu)=\mu-\mu^{2}/4$", xy=(3.6, d(3.6)),
                xytext=(2.6, -1.35), color=INK, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.0))

    for m in (1.0, 3.0):                        # kinks of the envelope
        ax.plot([m], [min(L(m, 0), L(m, 1), L(m, 2))], "o", ms=7,
                color=ACCENT, mec="white", mew=1.4, zorder=5)

    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$L(\mu,x)=x^{2}+\mu(1-x)$")
    ax.set_title(r"Q3(b)  For fixed $x$, $L$ is a straight line in $\mu$;"
                 "\n" r"the lower envelope of such lines is always concave",
                 pad=10)
    ax.set_xlim(0, 5.45)
    ax.set_ylim(-1.8, 5.2)
    tidy(ax)
    save(fig, "q3b_envelope.pdf")


if __name__ == "__main__":
    fig_q1a()
    fig_q1b()
    fig_q1c()
    fig_q2a()
    fig_q3b()
