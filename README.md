# autocall-monte-carlo
Pricing d'un Autocall sur CAC 40 par simulation Monte-Carlo (Python)


# Code Python


import yfinance as yf
import pandas as pd

# Télécharger les données automatiquement de Yahoo Finance du CAC 40#

ticker = "^FCHI"
data = yf.download(ticker, "2022-01-01", "2025-01-01")


print(data.head())
print(data.tail())

# Extraire le Close

close  = data["Close"]["^FCHI"]

# Calculer les rendements journaliers arithmétiques

returns = close.pct_change().dropna()

# Volatilité historique annualisée

vol_historique = returns.std() * (252 ** 0.5)

# Spot initial

S0 = close.iloc[-1]

print(f"Spot initial S0 : {S0:.2f}")
print(f"Volatilité historique annualisée : {vol_historique:.2%}")


import numpy as np

# Paramètres
r   = 0.03
vol = vol_historique
T   = 3               # maturité 3 ans
N   = 252 * T
M   = 10000           # nb de simulations
dt  = T / N

# Simulation
np.random.seed(42)
Z = np.random.standard_normal((M, N))
increments = (r - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * Z
trajectoires = S0 * np.exp(np.cumsum(increments, axis=1))

print(f"Shape des trajectoires : {trajectoires.shape}")
print(f"Valeur moyenne finale : {trajectoires[:, -1].mean():.2f}")
print(f"Valeur théorique E[S_T] = S0*e^(rT) : {S0 * np.exp(r * T):.2f}")

# Paramètres produit
nominal         = 1000
barriere_rappel = 1.00   # 100% de S0
barriere_capital= 0.60   # 60% de S0
coupon          = 0.10   # 10% par an

# Dates d'observation annuelles
dates_obs = [252, 504, 756]  # an 1, an 2, an 3

# Extraire les prix aux dates d'observation
spots_obs = trajectoires[:, [d-1 for d in dates_obs]]  # shape (10000, 3)
niveaux   = spots_obs / S0  # normaliser par S0 → on compare à 1.0

print(f"Shape spots_obs : {spots_obs.shape}")
print(f"% de trajectoires au-dessus de la barrière à 1 an : {(niveaux[:,0] >= barriere_rappel).mean():.2%}")
print(f"% de trajectoires au-dessus de la barrière à 2 ans : {(niveaux[:,1] >= barriere_rappel).mean():.2%}")
print(f"% de trajectoires au-dessus de la barrière à 3 ans : {(niveaux[:,2] >= barriere_rappel).mean():.2%}")


payoffs = np.zeros(M)  # payoff pour chaque simulation

for i in range(M):
    rappele = False

for j, t in enumerate([1, 2, 3]):  # an 1, 2, 3
        if niveaux[i, j] >= barriere_rappel:
            # Rappel automatique : capital + coupon × années écoulées
            payoffs[i] = nominal * (1 + coupon * t)
            rappele = True
            break  # on sort : le produit est terminé

if not rappele:
        # À maturité : pas de rappel
        if niveaux[i, -1] >= barriere_capital:
            # Au-dessus de la barrière de protection → capital remboursé
            payoffs[i] = nominal
else:
        # En dessous → perte proportionnelle à la baisse
            payoffs[i] = nominal * niveaux[i, -1]

# Statistiques
print(f"Payoff moyen         : {payoffs.mean():.2f}")
print(f"Payoff min           : {payoffs.min():.2f}")
print(f"Payoff max           : {payoffs.max():.2f}")
print(f"% de rappel total    : {(payoffs > nominal).mean():.2%}")
print(f"% de perte en capital: {(payoffs < nominal).mean():.2%}")

# Actualisation : prix = E[payoff] × e^(-rT_moyen)


prix_actualises = np.zeros(M)

for i in range(M):
    rappele = False
    for j, t in enumerate([1, 2, 3]):
        if niveaux[i, j] >= barriere_rappel:
            prix_actualises[i] = nominal * (1 + coupon * t) * np.exp(-r * t)
            rappele = True
            break
    if not rappele:
        if niveaux[i, -1] >= barriere_capital:
            prix_actualises[i] = nominal * np.exp(-r * 3)
        else:
            prix_actualises[i] = nominal * niveaux[i, -1] * np.exp(-r * 3)

prix = prix_actualises.mean()
print(f"Prix du produit      : {prix:.2f} EUR")
print(f"Prix en % du nominal : {prix/nominal:.2%}")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(14, 10))
fig.suptitle("Pricer Monte-Carlo — Autocall CAC 40",
             fontsize=14, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

# ── Graphique 1 : 50 trajectoires simulées ──────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
t_jours = np.linspace(0, 3, 756)
for i in range(50):
    ax1.plot(t_jours, trajectoires[i] / S0,
             alpha=0.3, linewidth=0.7, color='steelblue')
ax1.axhline(barriere_rappel,  color='green',  linestyle='--',
            linewidth=1.5, label='Barrière rappel (100%)')
ax1.axhline(barriere_capital, color='red',    linestyle='--',
            linewidth=1.5, label='Barrière capital (60%)')
ax1.axvline(1, color='grey', linestyle=':', linewidth=1)
ax1.axvline(2, color='grey', linestyle=':', linewidth=1)
ax1.axvline(3, color='grey', linestyle=':', linewidth=1)
ax1.set_title("50 trajectoires simulées")
ax1.set_xlabel("Temps (années)")
ax1.set_ylabel("S(t) / S0")
ax1.legend(fontsize=7)

# ── Graphique 2 : Distribution des payoffs ──────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(payoffs, bins=60, color='steelblue',
         edgecolor='white', linewidth=0.4)
ax2.axvline(nominal, color='orange', linestyle='--',
            linewidth=2, label='Nominal (1000€)')
ax2.axvline(payoffs.mean(), color='red', linestyle='-',
            linewidth=2, label=f'Moyenne ({payoffs.mean():.0f}€)')
ax2.set_title("Distribution des payoffs")
ax2.set_xlabel("Payoff (€)")
ax2.set_ylabel("Fréquence")
ax2.legend(fontsize=8)

# ── Graphique 3 : Date de rappel ────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
dates_rappel = []
for i in range(M):
    for j, t in enumerate([1, 2, 3]):
        if niveaux[i, j] >= barriere_rappel:
            dates_rappel.append(t)
            break

non_rappele = M - len(dates_rappel)
labels  = ['Rappel an 1', 'Rappel an 2', 'Rappel an 3', 'Non rappelé']
counts  = [dates_rappel.count(1), dates_rappel.count(2),
           dates_rappel.count(3), non_rappele]
colors  = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
ax3.bar(labels, counts, color=colors, edgecolor='white')
ax3.set_title("Scénarios de rappel")
ax3.set_ylabel("Nombre de simulations")
for k, v in enumerate(counts):
    ax3.text(k, v + 50, f'{v/M:.1%}', ha='center', fontsize=9)

# ── Graphique 4 : Prix vs Volatilité ────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
vols_range = np.linspace(0.10, 0.40, 15)
prix_vols  = []

for v in vols_range:
    inc = (r - 0.5*v**2)*dt + v*np.sqrt(dt)*Z
    traj_v = S0 * np.exp(np.cumsum(inc, axis=1))
    niv_v  = traj_v[:, [d-1 for d in dates_obs]] / S0
    px = np.zeros(M)
    for i in range(M):
        for j, t in enumerate([1, 2, 3]):
            if niv_v[i, j] >= barriere_rappel:
                px[i] = nominal*(1+coupon*t)*np.exp(-r*t)
                break
        else:
            if niv_v[i, -1] >= barriere_capital:
                px[i] = nominal*np.exp(-r*3)
            else:
                px[i] = nominal*niv_v[i,-1]*np.exp(-r*3)
    prix_vols.append(px.mean())

ax4.plot(vols_range*100, prix_vols, 'o-',
         color='steelblue', linewidth=2, markersize=5)
ax4.axhline(nominal, color='orange', linestyle='--', linewidth=1.5)
ax4.axvline(vol_historique*100, color='red', linestyle=':',
            linewidth=1.5, label=f'Vol actuelle ({vol_historique:.1%})')
ax4.set_title("Prix du produit vs Volatilité")
ax4.set_xlabel("Volatilité (%)")
ax4.set_ylabel("Prix (€)")
ax4.legend(fontsize=8)

plt.savefig("autocall_pricer.png", dpi=150, bbox_inches='tight')
plt.show()
print("Graphiques sauvegardés !")
