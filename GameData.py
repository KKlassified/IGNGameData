import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import operator
import math

df = pd.read_csv('C:\\Users\\kyle\\Desktop\\ign.csv', nrows=20000)# read CSV


def average(df):
    total = df['score'].sum()
    count = df.score.count()
    avg = total/count
    return avg


def score_chart_name(df):
    df1 = df.drop(columns=['platform', 'release_day', 'release_month', 'release_year', 'Unnamed: 0', 'genre','editors_choice','url']).drop_duplicates(subset='title', keep='first').reset_index()
    row = 0
    for x in df1.title:
        short = x.split()
        new = short[0] # + " " + short[1]
        df1.at[row, 'title'] = new
        row = row + 1
    avg = average(df1)
    print(avg)
    df1.plot.barh(x='title', y='score', rot=0, legend=True)
    plt.show()


def score_hist(df):
    df1 = df.drop(columns=['platform', 'release_day', 'release_month', 'release_year', 'Unnamed: 0', 'genre', 'editors_choice','url']).reset_index()
    plt.hist(df1.score, bins=10, histtype='bar', rwidth=1,color='r')
    plt.xlabel('Score')
    plt.ylabel('Number of Games')
    plt.title("Score Breakdown", size=22, family='Arial')
    plt.show()


def pie_chart_score_phrase(df):
    df1 = df.score_phrase.value_counts(normalize=True)
    size = []
    other = 0
    labels = []
    row = 0
    for x in df1:
        size.append(x)
        labels.append(df1.index[row])
        row = row + 1
    for x in size:
        if x < .12:
            other = other + x
    labels = labels[:6]
    labels.append('Other')
    size = size[:6]
    size.append(other)

    fig1, ax1 = plt.subplots()
    ax1.set_title('Score Phrase Breakdown', size=22)
    ax1.pie(size, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()


def pie_chart_seasonal_score(df):
    # count of all quarters
    q1 = 0
    q2 = 0
    q3 = 0
    q4 = 0
    # set size and labels of chart
    size = []
    labels = ['Quarter 1', 'Quarter 2', 'Quarter 3', 'Quarter 4']
    # place the values of release month in containers
    for x in df['release_month']:
        if x <= 3:
            q1 = q1 + 1
        if x <= 6:
            q2 = q2 + 1
        if x <= 9:
            q3 = q3 + 1
        if x <= 12:
            q4 = q4 + 1
    # set the sizes of each slice
    size.append(q1)
    size.append(q2)
    size.append(q3)
    size.append(q4)

    explode = (0, 0, 0, 0)

    fig1, ax1 = plt.subplots()
    ax1.set_title('Season Breakdown')
    ax1.pie(size, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.1f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct

def score_total_by_platform(df):
    unique = df.platform.unique() # unique values from platform column
    array = np.zeros(len(unique), dtype=int) # empty array
    array2 = np.zeros(len(unique), dtype=int)
    dict1 = dict(zip(unique, array)) # zip the two lists together
    dict2 = dict(zip(unique, array2))
    row = 0
    for x in df['score']: #l oop through all scores
        platform = df.at[row, 'platform'] # find platform for that score
        new_var = dict1[platform] # find that platform in list
        score_total = new_var + x # add score to the score saved for that platform
        row = row + 1 # go to next row
        dict1[platform] = score_total # update value for that platform
    row = 0
    for y in df['platform']: # same as above but counts time platform comes up
        platform = df.at[row, 'platform']
        count = dict2[y]
        new_count = count + 1
        dict2[platform] = new_count
        row = row + 1

    sorted_dict1 = dict(sorted(dict1.items(), key=operator.itemgetter(1), reverse=True)) # sort dict to put highest value first
    sorted_dict2 = dict(sorted(dict2.items(), key=operator.itemgetter(1), reverse=True))


# Pie chart for score of games by platform
# information is variable, size needs to be mutable
    labels = []
    size = []
    cat = 0
    sum = 0
    for i in sorted_dict1:
        if cat <= 4: # 4 is number of categories you want
            labels.append(i)
            score_totals = sorted_dict1[i]
            size.append(score_totals)
            cat = cat + 1
            sum = sum + score_totals

    explode = np.zeros(len(labels)) # make a list of all zeros
    explode[0] = (explode[0] + .1) # make first value which is highest in this case pop out

    fig3 = plt.figure(1, figsize=(20, 10))
    chart_1 = fig3.add_subplot(121)
    chart_2 = fig3.add_subplot(122)
    chart_1.pie(size, labels=labels, autopct=make_autopct(size),
                shadow=True, startangle=90, textprops={'fontsize': 11}, wedgeprops={'edgecolor': 'black'})
    my_circle = plt.Circle((0, 0), 0.7, color='white')
    p = plt.gcf()
    p.gca().add_artist(my_circle)
    chart_2.pie(size, explode=explode, labels=labels, autopct=make_autopct(size),
                shadow=True, startangle=90, textprops={'fontsize': 11}, wedgeprops={'edgecolor': 'black'})
    plt.show()

# pie chart of number of games counted

    labels = []
    size = []
    cat = 0
    sum = 0
    for i in sorted_dict2:
        if cat <= 4:  # 4 is number of categories you want
            labels.append(i)
            score_totals = sorted_dict2[i]
            size.append(score_totals)
            cat = cat + 1
            sum = sum + score_totals

    explode = np.zeros(len(labels))  # make a list of all zeros
    explode[0] = (explode[0] + .1)  # make first value which is highest in this case pop out

    fig3 = plt.figure(1, figsize=(20, 10))
    chart_1 = fig3.add_subplot(121)
    chart_2 = fig3.add_subplot(122)
    chart_1.pie(size, labels=labels, autopct=make_autopct(size),
            shadow=True, startangle=90, textprops={'fontsize': 11}, wedgeprops={'edgecolor': 'black'})
    my_circle = plt.Circle((0, 0), 0.7, color='white')
    p = plt.gcf()
    p.gca().add_artist(my_circle)
    chart_2.pie(size, explode=explode, labels=labels, autopct=make_autopct(size),
            shadow=True, startangle=90, textprops={'fontsize': 11}, wedgeprops={'edgecolor': 'black'})
    plt.show()


def editors_choice(df):
    choice = [0,0] # set choices either yes or no
    labels = ['Yes','No'] # set label
    for x in df['editors_choice']:
        if x.lower() == 'y': # sort either yes or no
            choice[0] = choice[0] + 1
        else:
            choice[1] = choice[1] + 1
    # create pie chart
    fig2, ax2 = plt.subplots()
    colors = ['#b80f0a', '#32CD32'] # set colors red and green
    ax2.pie(choice, labels=labels, autopct=make_autopct(choice),
            shadow=True, startangle=90, textprops={'fontsize': 11}, wedgeprops={'edgecolor': 'black'}, colors= colors)
    ax2.axis('equal')  # Equal aspect ratio ensures that pie is dr
    plt.show()

def genre_breakdown(df):
    unique = df.genre.unique() # unique values from genre column
    print(unique)

def stackplot():
    row = 0
    scores = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # set score values for x axis
    array = np.zeros(len(scores), dtype=int) # create arrays to start all scores at 0
    dict1 = dict(zip(scores, array)) # zip together scores and values for scores
    array2 = np.zeros(len(scores), dtype=int)
    dict2 = dict(zip(scores, array2))
    array3 = np.zeros(len(scores), dtype=int)
    dict3 = dict(zip(scores, array3))
    for x in df['score']: # iterate through each score
        platform = df.at[row, 'platform'] # find platform for that score
        if platform == 'PlayStation 2': # check what platform
            rounded_score = int(x) # round score so it fits in our catgories
            count = dict1[rounded_score] # find that number in our dict and find its value
            new_count = count + 1 # add one to value for new game
            dict1[rounded_score] = new_count # update value in dict

        elif platform == 'PlayStation 3':
            rounded_score = int(x)
            count = dict2[rounded_score]
            new_count = count + 1
            dict2[rounded_score] = new_count

        elif platform == 'PlayStation 4':
            rounded_score = int(x)
            count = dict3[rounded_score]
            new_count = count + 1
            dict3[rounded_score] = new_count

        row = row + 1

    y1 = [dict1[0], dict1[1], dict1[2], dict1[3], dict1[4], dict1[5], dict1[6], dict1[7], dict1[8], dict1[9], dict1[10]]  # set y values from the values in dict
    y2 = [dict2[0], dict2[1], dict2[2], dict2[3], dict2[4], dict2[5], dict2[6], dict2[7], dict2[8], dict2[9], dict2[10]]
    y3 = [dict3[0], dict3[1], dict3[2], dict3[3], dict3[4], dict3[5], dict3[6], dict3[7], dict3[8], dict3[9], dict3[10]]

    labels = ["PlayStation 2 ", "PlayStation 3", "PlayStation 4"]

    fig, ax = plt.subplots()
    ax.stackplot(scores, y1, y2, y3, labels=labels)
    ax.legend(loc='upper left')
    plt.show()

    plt.plot(scores, y1, 'r--', scores,y2, 'bo', scores, y3, 'g-')
    plt.legend(labels)
    plt.xlabel('Score')
    plt.ylabel('Number of games')
    plt.show()




pie_chart_seasonal_score(df)
pie_chart_score_phrase(df)
score_hist(df)
score_total_by_platform(df)
editors_choice(df)
genre_breakdown(df)
stackplot()
