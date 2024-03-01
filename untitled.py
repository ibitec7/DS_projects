import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = '/home/ibrahim/Downloads/data/Electric_Vehicle_Population_Data.csv'
df = pd.read_csv(file_path)

plt.style.use('ggplot')

df.head()

import my_func as mf
df.columns = mf.df_col(df.columns.tolist(),20)

# df_col(df.columns.to_list(),15)
col_list=df.columns.to_list()
col_list

df.info()

df.describe()

df = df.dropna()
df.info()

df['dummy'] = 1

corr_mat = df.corr()
sns.heatmap(corr_mat ,annot = True,fmt='.2f',cmap='RdBu_r')
plt.title('Correlation Matrix of Numeric Columns in DataFramewhy is iloc not working in dask')

df_ev = df[df.evt.isin(['Battery Electric Vehicle (BEV)'])]
print(df_ev[df_ev.electric_range==0].count())

df_ev2 = df_ev[df_ev['cafve']=='Clean Alternative Fuel Vehicle Eligible']

print(df_ev2[df_ev2['electric_range']==0].count())

avg_rng_mk = df_ev2.groupby('make')['electric_range'].agg('mean')
ev_make = df_ev2['make'].sort_values(ascending=True).unique()
plt.bar(ev_make,avg_rng_mk,color='b',alpha=0.7)
plt.xlabel('Make')
plt.ylabel("Mean Range(km)")
plt.title('Avg Range by Make(EV)')
plt.xticks(rotation = 90)
plt.show()

sns.boxplot(data=df_ev2,x='electric_range',y='make',color='b')
plt.figure(figsize=(16,10))

ev_count = df_ev['make'].value_counts()
ev_count.plot(kind='bar',color='b',alpha=0.7,logy=True)

ev_count

evt_count = pd.pivot_table(data=df,columns='evt',values='dummy',aggfunc='sum',index='make')
evt_count.plot(kind='bar',logy=True,figsize=(10,6))

ev_count

print(avg_rng_mk)

year = df_ev['model_year'].sort_values(ascending=True).unique()
avg_rng_yr = df_ev.groupby('model_year')['electric_range'].agg('mean')

plt.scatter(year, avg_rng_yr, linestyle = 'dotted', marker = 'x', color = 'blue')
plt.xlabel('Model Year')
plt.ylabel('Range(km)')
plt.title('Range by Model Year (EV)')
plt.show()

sns.kdeplot(data=df_ev['electric_range'],multiple='layer',c='blue')
plt.title('Electric Range distribution (KDE)')
plt.xlabel('Electric Range(km)')

plt.hist(x=df_ev2['electric_range'],bins=40,color='b',alpha=0.7)
plt.title('Electric Range Distribution (Histogram)')
plt.show()

corr_ev = df_ev.corr()
sns.heatmap(corr_ev,annot=True,fmt='.2f')

rng_make_yr = pd.pivot_table(data=df_ev,values='electric_range',columns='model_year',index='make',aggfunc='mean')
rng_make_yr

df_ev.head()

df_ev.state.unique()

mk_county = pd.pivot_table(data=df_ev,columns='make',index='county',values='dummy',aggfunc='count')
mk_county

col_list

ev_share=df_ev.make.value_counts(normalize=True,sort=True)

ev_share

col = df_ev['make'].unique()
share = list()
for i in col:
    share.append(df_ev[df_ev['make']==i]['make'].count())
total = sum(share)
share /= total
ev_share = pd.DataFrame({'make':col,'share':share})

others = {'make':'OTHERS','share':ev_share[ev_share['share']<0.03]['share'].sum()}
ev_share.loc[len(df_ev)] = others

ev_share.drop(inplace=True,index=ev_share[ev_share['share']<0.03].index)

ev_share.set_index('make',inplace=True)
ev_share.index

ev_share['share'].plot(kind='pie')
plt.title('Market Share')