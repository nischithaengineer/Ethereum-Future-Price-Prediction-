from flask import Flask, render_template, request
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from model import future,base, create_steps

app = Flask(__name__)

df = pd.read_csv(f'{base}/ETH_1H.csv', parse_dates=['Date'], index_col=['Date'], dayfirst=True, date_format='%d-%m-%Y %H:%M')
df=df.sort_index()

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        to_date = request.form['td']
        steps = create_steps(to_date)
        new = future(steps)
        out = np.append(df.Close.values, new)
        plt.switch_backend('Agg')
        plt.figure(figsize=(20, 2))
        plt.figure(facecolor='black', dpi=300)
        ax = plt.axes()
        ax.set_facecolor("black")
        plt.plot(out[30000:], linewidth=0.5, color='red', label='Future')
        plt.plot(df.Close.values[30000:], linewidth=0.5, color='white', label='Past')
        plt.legend()
        plt.savefig(f'{base}/static/output.png')
    else:
        out = df.Close.values  # Initial rendering, no prediction

    return render_template('Index.html')

if __name__ == "__main__":
    app.run(debug=True)
