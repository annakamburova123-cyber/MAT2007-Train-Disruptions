
##Project.py - Seasonal patterns in Dutch train disruptions

##Research question:
    #Are train disruptions in the Netherlands worse in some seasons than others?

##Data:   Rijden de Treinen open data (CC BY 4.0), years 2022-2025
         #https://www.rijdendetreinen.nl/en/open-data/disruptions


from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##Settings--------------------------------------------------------------------------##

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"

# Post-2017 reporting system, excluding the COVID years 2020-2021.
YEARS = [2022, 2023, 2024, 2025]

# Disruptions longer than this are treated as long-term events.
CUTOFF_HOURS = 24

# Meteorological seasons (KNMI convention).
SEASON_OF_MONTH = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}
SEASONS = ["Winter", "Spring", "Summer", "Autumn"]

# Colourblind-safe colours (Paul Tol's palettes).
SEASON_COLOURS = {"Winter": "#0077BB", "Spring": "#009988",
                  "Summer": "#EE7733", "Autumn": "#882255"}
ALL_COLOUR = "#B8C2CC"
SHORT_COLOUR = "#0077BB"
TEXT_GREY = "#333333"


##Load and clean--------------------------------------------------------------------------##

def load_data():
    #Read one CSV file per year and combine them into one table
    frames = []
    for year in YEARS:
        path = DATA_DIR / f"disruptions-{year}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Download it from "
                "https://www.rijdendetreinen.nl/en/open-data/disruptions"
            )
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def clean_data(df):
    #Remove duplicates and rows without a valid start time. add seasons
    n_start = len(df)
    df = df.drop_duplicates(subset="rdt_id")
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df = df.dropna(subset=["start_time"])
    df = df[df["start_time"].dt.year.isin(YEARS)].copy()
    df["season"] = df["start_time"].dt.month.map(SEASON_OF_MONTH)
    print(f"Kept {len(df)} of {n_start} disruptions after cleaning.")
    return df


def days_per_season():
    #Number of calendar days in each season over all selected years
    days = pd.date_range(f"{min(YEARS)}-01-01", f"{max(YEARS)}-12-31", freq="D")
    days = days[days.year.isin(YEARS)]
    return pd.Series(days.month).map(SEASON_OF_MONTH).value_counts().reindex(SEASONS)


##Analysis--------------------------------------------------------------------------##

def frequency_table(df, days):
    #Disruptions per day. Poisson uncertainty: sigma = sqrt(N) / days
    n = df["season"].value_counts().reindex(SEASONS, fill_value=0)
    return pd.DataFrame({
        "N": n,
        "per_day": n / days,
        "per_day_err": np.sqrt(n) / days,
    })


def duration_table(df, days):
   
    #Hours of disruption per day
    #Uncertainty: sigma = sqrt(sum of each duration squared) / days,
    #i.e. sqrt(N) where each disruption is weighted by its length.
    
    hours = df.dropna(subset=["duration_minutes"]).assign(
        hours=lambda d: d["duration_minutes"] / 60
    ).groupby("season")["hours"]
    table = pd.DataFrame({
        "total_hours": hours.sum(),
        "median_hours": hours.median(),
        "per_day": hours.sum() / days,
        "per_day_err": np.sqrt(hours.apply(lambda h: (h ** 2).sum())) / days,
    })
    return table.reindex(SEASONS)


def compare_extremes(table, label):
    #Difference between the highest and lowest season, and its significance
    high, low = table["per_day"].idxmax(), table["per_day"].idxmin()
    diff = table.loc[high, "per_day"] - table.loc[low, "per_day"]
    err = np.hypot(table.loc[high, "per_day_err"], table.loc[low, "per_day_err"])
    print(f"{label}: {high} - {low} = {diff:.2f} +/- {err:.2f} "
          f"({diff / err:.1f} sigma)")


##Plots--------------------------------------------------------------------------##

def set_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlepad": 12,
        "axes.labelcolor": TEXT_GREY, "axes.edgecolor": "#BBBBBB",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
        "grid.color": "#E5E5E5",
        "xtick.color": TEXT_GREY, "ytick.color": TEXT_GREY,
        "xtick.bottom": False,
        "legend.frameon": False,
        "savefig.bbox": "tight",
    })


def save(fig, name):
    ##PNG for the slides, PDF for the LaTeX report since its a bit sharper
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=300)
    fig.savefig(OUTPUT_DIR / f"{name}.pdf")
    plt.close(fig)


def label_bars(ax, bars, values, errors, top):
    #Write each value just above its error bar
    for bar, v, e in zip(bars, values, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, v + e + top * 0.015,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=TEXT_GREY)

#frequency plot
def plot_frequency(freq):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    top = (freq["per_day"] + freq["per_day_err"]).max()
    bars = ax.bar(SEASONS, freq["per_day"], yerr=freq["per_day_err"], width=0.6,
                  color=[SEASON_COLOURS[s] for s in SEASONS], capsize=5,
                  error_kw={"ecolor": TEXT_GREY})
    label_bars(ax, bars, freq["per_day"], freq["per_day_err"], top)
    ax.set_ylim(0, top * 1.12)
    ax.set_ylabel("Disruptions per day")
    ax.set_title("Train disruptions per day by season")
    save(fig, "disruptions_per_season")


def plot_distribution(df):
    ##Box plot of durations on a log axis (minutes to  month)
    d = df[df["duration_minutes"] > 0]
    data = [d.loc[d["season"] == s, "duration_minutes"] for s in SEASONS]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    box = ax.boxplot(
        data, tick_labels=SEASONS, widths=0.55, patch_artist=True,
        medianprops={"color": "white", "linewidth": 2},
        whiskerprops={"color": TEXT_GREY}, capprops={"color": TEXT_GREY},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.25,
                    "markeredgecolor": "none"},
    )
    for patch, flier, s in zip(box["boxes"], box["fliers"], SEASONS):
        patch.set(facecolor=SEASON_COLOURS[s], edgecolor=SEASON_COLOURS[s])
        flier.set_markerfacecolor(SEASON_COLOURS[s])
    ax.set_yscale("log")
    ticks = [1, 10, 60, 360, 1440, 10080, 43200]  # in minutes
    ax.set_yticks(ticks, ["1 min", "10 min", "1 h", "6 h", "1 day", "1 week", "1 month"])
    ax.minorticks_off()
    ax.set_ylabel("Duration of disruption")
    ax.set_title("Distribution of disruption durations by season")
    save(fig, "duration_distribution")


def plot_comparison(dur_all, dur_short):
    #Hours per day: all disruptions vs. only those up to the cutoff (24 hrs)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(SEASONS))
    width = 0.38
    top = max((t["per_day"] + t["per_day_err"]).max() for t in (dur_all, dur_short))
    for table, colour, label, offset in [
        (dur_all, ALL_COLOUR, "All disruptions", -width / 2),
        (dur_short, SHORT_COLOUR, f"Only disruptions \u2264 {CUTOFF_HOURS} h", width / 2),
    ]:
        bars = ax.bar(x + offset, table["per_day"], width, yerr=table["per_day_err"],
                      color=colour, label=label, capsize=4,
                      error_kw={"ecolor": TEXT_GREY})
        label_bars(ax, bars, table["per_day"], table["per_day_err"], top)
    ax.set_xticks(x, SEASONS)
    ax.set_ylim(0, top * 1.15)
    ax.set_ylabel("Hours of disruption per day")
    ax.set_title("Disruption time per day: all vs. short disruptions")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    save(fig, "duration_comparison")


##Run--------------------------------------------------------------------------##

#This should run all the data!
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    set_style()

    df = clean_data(load_data())
    days = days_per_season()
    short = df[df["duration_minutes"] <= CUTOFF_HOURS * 60]

    freq = frequency_table(df, days)
    dur_all = duration_table(df, days)
    dur_short = duration_table(short, days)

    print("\nDisruptions per day:")
    print(freq.round(2))
    print("\nHours of disruption per day (all disruptions):")
    print(dur_all.round(2))
    print(f"\nHours of disruption per day (only <= {CUTOFF_HOURS} h):")
    print(dur_short.round(2))

    print()
    compare_extremes(freq, "Disruptions per day")
    compare_extremes(dur_all, "Hours per day (all)")
    compare_extremes(dur_short, f"Hours per day (<= {CUTOFF_HOURS} h)")

    print("\nTen longest winter disruptions:")
    longest = df[df["season"] == "Winter"].nlargest(10, "duration_minutes")
    print(longest[["start_time", "duration_minutes", "cause_en"]].to_string(index=False))

    freq.round(3).to_csv(OUTPUT_DIR / "frequency.csv")
    dur_all.round(3).to_csv(OUTPUT_DIR / "duration_all.csv")
    dur_short.round(3).to_csv(OUTPUT_DIR / "duration_short.csv")

    plot_frequency(freq)
    plot_distribution(df)
    plot_comparison(dur_all, dur_short)
    print(f"\nResults saved in '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()