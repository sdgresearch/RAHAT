import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import os
import re
from matplotlib import cm
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Configuration: Folder where your files are located
DATA_FOLDER = r'C:/Users/ytuli/Project/RAHAT/RAHAT/Data Logger_Jodhpur-20250325T181905Z-001/Data Logger_Jodhpur/house hold temperature/'
sname = r"C:/Users/ytuli/Project/RAHAT/RAHAT/boxplot.png"

# Outdoor file (expects columns: date, averageT, minT, maxT)
OUTDOOR_FILE = r"C:/Users/ytuli/Project/RAHAT/RAHAT/outdoorT.xlsx"


def read_datalogger_file(filepath):
    filename = os.path.basename(filepath)
    house_name = os.path.splitext(filename)[0]

    if house_name.endswith('PYes'):
        paint = 'PYes'
    elif house_name.endswith('PNo'):
        paint = 'PNo'
    else:
        paint = 'Unknown'

    ext = os.path.splitext(filepath)[1].lower()
    df = None
    header_row_index = None

    if ext in ['.xls', '.xlsx']:
        try:
            preview_df = pd.read_excel(filepath, header=None, nrows=30)
            for i, row in preview_df.iterrows():
                row_str = row.astype(str).str.lower().tolist()
                if any('datetime' in s for s in row_str) and any('ti' in s for s in row_str):
                    header_row_index = i
                    break
            if header_row_index is not None:
                df = pd.read_excel(filepath, header=header_row_index)
        except Exception:
            pass

    if df is None:
        try:
            header_row_index = 0
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i > 30:
                        break
                    if ("Datetime" in line) and (("TI" in line) or ("NO" in line)):
                        header_row_index = i
                        break
            df = pd.read_csv(filepath, header=header_row_index, sep=None, engine='python')
        except Exception:
            return None

    if df is None or df.empty:
        return None

    df.columns = [str(c).strip() for c in df.columns]
    needed = ['Datetime', 'RH', 'TI']
    if not all(k in df.columns for k in needed):
        return None

    df = df[needed].copy()
    df['Datetime'] = pd.to_datetime(df['Datetime'], dayfirst=True, errors='coerce')
    df['RH'] = pd.to_numeric(df['RH'], errors='coerce')
    df['TI'] = pd.to_numeric(df['TI'], errors='coerce')
    df = df.dropna(subset=['Datetime'])
    df['House'] = house_name
    df['Paint'] = paint

    return df


def read_outdoor_file(outdoor_path):
    """
    Reads outdoorT.xlsx with columns: date, averageT, minT, maxT
    Returns a dataframe indexed by python date (datetime.date)
    """
    if outdoor_path is None or (not os.path.exists(outdoor_path)):
        print(f"Outdoor file not found: {outdoor_path}")
        return None

    odf = pd.read_excel(outdoor_path).copy()

    # Be a bit robust to column casing/whitespace
    odf.columns = [str(c).strip() for c in odf.columns]

    required = ['date', 'averageT', 'minT', 'maxT']
    if not all(c in odf.columns for c in required):
        print(f"Outdoor file must contain columns: {required}. Found: {list(odf.columns)}")
        return None

    odf['date'] = pd.to_datetime(odf['date'], errors='coerce').dt.date
    odf['averageT'] = pd.to_numeric(odf['averageT'], errors='coerce')
    odf['minT'] = pd.to_numeric(odf['minT'], errors='coerce')
    odf['maxT'] = pd.to_numeric(odf['maxT'], errors='coerce')
    odf = odf.dropna(subset=['date'])
    odf = odf.set_index('date').sort_index()

    return odf


def get_color(house):
    match = re.search(r'[Ss](\d+)', house)
    if not match:
        return 'black'
    num = int(match.group(1))

    if 1 <= num <= 20:
        cmap = cm.Reds
        idx = (num - 1) / 19
    elif 21 <= num <= 40:
        cmap = cm.Greens
        idx = (num - 21) / 19
    elif 41 <= num <= 60:
        cmap = cm.Blues
        idx = (num - 41) / 19
    elif 61 <= num <= 80:
        cmap = cm.Oranges
        idx = (num - 61) / 19
    else:
        return 'black'

    idx = max(0, min(1, idx))
    return cmap(idx)


def plot_timeseries(combined_df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

    houses = sorted(
        combined_df['House'].unique(),
        key=lambda h: int(re.search(r'[Ss](\d+)', h).group(1)) if re.search(r'[Ss](\d+)', h) else 9999
    )

    for h in houses:
        d = combined_df[combined_df['House'] == h].sort_values('Datetime')
        c = get_color(h)
        ax1.plot(d['Datetime'], d['TI'], color=c, linewidth=1, alpha=0.8, label=h)
        ax2.plot(d['Datetime'], d['RH'], color=c, linewidth=1, alpha=0.8, label=h)

    ax1.set_ylabel("Temperature (°C)")
    ax2.set_ylabel("Relative Humidity (%)")
    ax1.grid(True, ls='--')
    ax2.grid(True, ls='--')
    ax2.set_ylim(0, 100)
    plt.xticks(rotation=45)
    ax1.legend(fontsize=5, ncol=5)
    plt.tight_layout()


def plot_boxplots_over_time(combined_df, outdoor_df=None):
    df = combined_df[combined_df['Paint'].isin(['PYes', 'PNo'])].copy()
    if df.empty:
        print("No paint data available.")
        return

    df['Date'] = pd.to_datetime(df['Datetime']).dt.date
    dates = sorted(df['Date'].unique())

    fig, (ax_ti, ax_rh) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    base = np.arange(len(dates))
    offset = 0.15
    print(outdoor_df)

    # ---- Add outdoor line plot + min-max band on Temperature subplot
    if outdoor_df is not None and (not outdoor_df.empty):
        out_avg, out_min, out_max = [], [], []
        for day in dates:
            if day in outdoor_df.index:
                out_avg.append(outdoor_df.loc[day, 'averageT'])
                out_min.append(outdoor_df.loc[day, 'minT'])
                out_max.append(outdoor_df.loc[day, 'maxT'])
            else:
                out_avg.append(np.nan)
                out_min.append(np.nan)
                out_max.append(np.nan)

        ax_ti.plot(base, out_avg, marker='o', linewidth=1.5, label='Outdoor mean T')
        ax_ti.fill_between(base, out_min, out_max, alpha=0.2, label='Outdoor min–max T')

    # ---- Boxplot settings:
    common_bp_kwargs = dict(
        widths=0.25,
        patch_artist=True,
        showfliers=False,
        showmeans=True,
        meanline=False,
        whis=(0, 100),
        medianprops=dict(color='black', linewidth=1.2),
        whiskerprops=dict(color='black', linewidth=1.0),
        capprops=dict(color='black', linewidth=1.0),
        meanprops=dict(marker='o', markerfacecolor='black', markeredgecolor='black', markersize=3)
    )

    for i, day in enumerate(dates):
        d = df[df['Date'] == day]
        yes = d[d['Paint'] == 'PYes']
        no = d[d['Paint'] == 'PNo']

        if len(yes) > 0:
            ax_ti.boxplot([yes['TI']], positions=[base[i] - offset],
                          boxprops=dict(facecolor='blue', alpha=0.6),
                          **common_bp_kwargs)

        if len(no) > 0:
            ax_ti.boxplot([no['TI']], positions=[base[i] + offset],
                          boxprops=dict(facecolor='red', alpha=0.6),
                          **common_bp_kwargs)

        if len(yes) > 0:
            ax_rh.boxplot([yes['RH']], positions=[base[i] - offset],
                          boxprops=dict(facecolor='blue', alpha=0.6),
                          **common_bp_kwargs)

        if len(no) > 0:
            ax_rh.boxplot([no['RH']], positions=[base[i] + offset],
                          boxprops=dict(facecolor='red', alpha=0.6),
                          **common_bp_kwargs)

    # --- CHANGED: Reduce tick frequency to one per week (every 7th day) ---
    tick_indices = base[::7]
    tick_labels = [pd.to_datetime(dates[i]).strftime('%d/%m') for i in tick_indices]

    ax_rh.set_xticks(tick_indices)
    ax_rh.set_xticklabels(tick_labels)
    # ---------------------------------------------------------------------

    ax_ti.set_ylabel("T(°C)")
    ax_rh.set_ylabel("RH(%)")
    ax_rh.set_ylim(0, 100)

    legend_handles = [
        Patch(facecolor='blue', alpha=0.6, label='Intervention'),
        Patch(facecolor='red', alpha=0.6, label='No Intervention'),
        Line2D([0], [0], color='C0', lw=1.5, marker='o', label='Outdoor'),
        Patch(facecolor='C0', alpha=0.2, label='Outdoor Range')
    ]

    ax_ti.legend(handles=legend_handles, loc='lower right', fontsize=9)

    ax_ti.grid(True, ls='--', alpha=0.5)
    ax_rh.grid(True, ls='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(sname, dpi=300)
    plt.show()


def main():
    files = []
    for ext in ['*.csv', '*.xls', '*.xlsx']:
        files += glob.glob(os.path.join(DATA_FOLDER, ext))

    files.sort(
        key=lambda p: int(re.search(r'[Ss](\d+)', os.path.basename(p)).group(1))
        if re.search(r'[Ss](\d+)', os.path.basename(p)) else 999999
    )

    dfs = []
    for f in files:
        d = read_datalogger_file(f)
        if d is not None:
            dfs.append(d)

    if not dfs:
        print("No valid data.")
        return

    df = pd.concat(dfs, ignore_index=True)

    outdoor_df = read_outdoor_file(OUTDOOR_FILE)
    plot_boxplots_over_time(df, outdoor_df=outdoor_df)


if __name__ == "__main__":
    main()