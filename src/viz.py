import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

def ep_by_ydline_and_season(years):
    df_list = []
    for y in years:
        df_list.append(pd.read_parquet('data/pbp/play_by_play_{yr}.parquet'.format(yr = y)))
    size = math.ceil(math.sqrt(len(df_list)))
    fig, axs = plt.subplots(nrows=size, ncols=size, figsize=(20, 12))
    axs_flat = axs.flatten()
    for i, (df, ax) in enumerate(zip(df_list, axs_flat)):
        sns.lineplot(data=df, x="yardline_100", y="ep", ax=ax, hue="down", palette = "bright")
        ax.set_title(str(years[i]) + " Season")
        ax.set_xlabel("Yards from Opponents Endzone")
        ax.set_ylabel("Expected Points")
    plt.title('Expected Points by Yardline by Season')
    plt.tight_layout()
    plt.savefig('visualizations/ExpectedPointsDown.png')
    plt.show()

def drive_outcome_by_yardline(years):
    df = pd.DataFrame()
    for y in years:
        new_df = pd.read_parquet('data/pbp/play_by_play_{yr}.parquet'.format(yr = y))
        df = pd.concat([df, new_df])
    plt.figure(figsize = (12,8))
    yd_bins = range(0,101,5)
    plot_df = df[df['down'].isin([1,2,3,4])].reset_index(drop = True)
    plot_df.groupby(pd.cut(plot_df['yardline_100'], bins=yd_bins))['drive_ended_with_score'].mean().plot(kind='bar')
    plt.title('Percentage of Drives Ending with Score by Yardline Bin')
    plt.xlabel('Yards to Go Bin')
    plt.ylabel('Percentage of Drives Ending with Score')
    plt.gca().invert_xaxis() 
    plt.tight_layout()
    plt.savefig('visualizations/DriveOutcomeByYardline.png')
    plt.show()