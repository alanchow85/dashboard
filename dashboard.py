import dash
from dash import dcc 
from dash import Input 
from dash import Output 
from dash import Dash
from dash import html
import pandas as pd
import numpy as np
import flask

import plotly.express as px
from dash.exceptions import PreventUpdate
from dash.dependencies import Output, Input  #important for the dependencies of multicallback
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor



# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 09:12:59 2018

@author: Alan


https://www.w3schools.com/css/css_inline-block.asp
for things on the display, can refer to w3s school on the display is side by side if we put style: {'display:'inline-block'} 

Dash converts Python classes into HTML
This conversion happens behind the scenes by Dash's JavaScript front-end
"""

data = pd.read_csv('resale-flat-prices-based-on-registration-date-from-jan-2017-onwards.csv')
data = data.sort_values(by='resale_price', ascending = False)

town = data['town']
flat_type = data['flat_type']
price = data['resale_price']


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']



server = flask.Flask(__name__)
app1 = Dash(__name__, external_stylesheets=external_stylesheets, server=server, url_base_pathname='/dashboard/')
app2 = Dash(__name__, external_stylesheets=external_stylesheets, server=server, url_base_pathname='/reports/')

# dashboard 1 in app1

app1.layout = html.Div([
    html.H1('Real-Time Singapore Housing Dashboard', style={'text-align': 'center'}),
    html.Div([
        html.P('Dropdown list to select by Town'),
    ]),
    
    html.Div([
        dcc.Dropdown(id='town-dropdown', options =[{'label': i, 'value': i }
                                                   for i in data['town'].unique()], value = 'BISHAN'
        ),
        html.P('Slider to select flats lesser than or equal to selected floor area'),
        dcc.Slider(
        data['floor_area_sqm'].min(),
        data['floor_area_sqm'].max(),
        step=None,
        value=data['floor_area_sqm'].max(),
        marks={str(floor_area): str(floor_area) for floor_area in data['floor_area_sqm'].unique()},
        id='floor_area-slider'
        ),
        
        dcc.Graph(id='price-graph',style={'display': 'inline-block'}, figure={'layout': {'title': 'Average price for each type of house','height': 400, 'width' : 900, 'xaxis':{'title': 'alan x-axis'}}}
        ),
        dcc.Graph(id='price-graph2',style={'display': 'inline-block'}, figure={'layout': {'title': 'Observation of floor space & year built affecting price', 'height': 420, 'width' : 900}}
        ),
        dcc.Graph(id='price-graph3',style={'display': 'inline-block'}, figure={'layout': {'title': 'Singapore HDB housing price from 2017 onwards', 'height': 400, 'width' : 900}}
        ),
        
        dcc.Graph(
            id='top-right-graph', style={'display': 'inline-block'},
            figure={
                 'data': [{
                         'z': price,
                         'x':town,
                         'y':flat_type,       
                         'type': 'heatmap'
                                 }],
                'layout': {
                    'title':'Heatmap of resale housing price at various towns', 
                    'height': 400,
                     'width' : 900,
                    'margin': {'l': 48, 'b': 25, 't': 70, 'r': 40}
                }
            }
        ),
    ]),
    
    html.Div([
        html.P('Hi all!'),
         
    ])
])

app2.layout = html.Div([
    html.H1('Real-Time Singapore Housing Dashboard2', style={'text-align': 'center'}),
    html.Div([
        html.P('Dropdown list to select by Town'),
    ]),
    
    html.Div([
        dcc.Dropdown(id='town-dropdown', options =[{'label': i, 'value': i }
                                                   for i in data['town'].unique()], value = 'BISHAN'
        ),
        html.P('Slider to select flats lesser than or equal to selected floor area'),
        dcc.Slider(
        data['floor_area_sqm'].min(),
        data['floor_area_sqm'].max(),
        step=None,
        value=data['floor_area_sqm'].max(),
        marks={str(floor_area): str(floor_area) for floor_area in data['floor_area_sqm'].unique()},
        id='floor_area-slider'
        ),
        
        dcc.Graph(id='price-graph4',style={'display': 'inline-block'}, figure={'layout': {'title': 'Average price for each type of house','height': 400, 'width' : 900, 'xaxis':{'title': 'alan x-axis'}}}
        ),
        dcc.Graph(id='price-graph5',style={'display': 'inline-block'}, figure={'layout': {'title': 'Observation of floor space & year built affecting price', 'height': 420, 'width' : 900}}
        ),
        dcc.Graph(id='price-graph6',style={'display': 'inline-block'}, figure={'layout': {'title': 'Machine learning model - K-mean Clustering', 'height': 420, 'width' : 900}}
        ),
        
        dcc.Graph(
            id='top-right-graph', style={'display': 'inline-block'},
            figure={
                 'data': [{
                         'z': price,
                         'x':town,
                         'y':flat_type,       
                         'type': 'heatmap'
                                 }],
                'layout': {
                    'title':'Heatmap of resale housing price at various towns', 
                    'height': 400,
                     'width' : 900,
                    'margin': {'l': 48, 'b': 25, 't': 70, 'r': 40}
                }
            }
        ),
    ]),
    
    html.Div([
        html.P('Hi all!'),
         
    ])
])


@app1.callback(
    [Output('price-graph', 'figure'),
    Output('price-graph2', 'figure'),
    Output('price-graph3', 'figure')],
    [Input('town-dropdown', 'value'), 
    Input('floor_area-slider', 'value')]
)

def multi_output(selected_town, floor_area):  #as you have more inputs, e.g 3, you will then have 3 variable in the brackets
    filtered_data = data[data.town == selected_town] #to filter only those that are same as the selected town
    filtered_data = data[data.floor_area_sqm < floor_area]
    
    flat_type = filtered_data['flat_type'].unique()
    average = []
    numTransaction = []
    for j in range(len(flat_type)):
        price = 0
        count = 0
        for i in range(len(filtered_data)):
            if flat_type[j] == filtered_data['flat_type'].iloc[i]:
                price = price + filtered_data['resale_price'].iloc[i]
                count = count + 1
        average.append(price/count)
        numTransaction.append(count)
    fig = px.bar(x=flat_type, y=average, labels={
        "x": "flat type",
        "y": "average price",
        "title": "Average price for each type of house"
        }
    )
    fig2 = px.scatter(x=filtered_data['lease_commence_date'], y=filtered_data['floor_area_sqm'], color=filtered_data['flat_type'], size=filtered_data['resale_price'], labels={"x": "Year built","y": "Floor space(sqm)"})
    fig3 = px.pie(labels=flat_type, values=numTransaction, names=flat_type)
    
    return fig, fig2, fig3

@app2.callback(
    [Output('price-graph4', 'figure'),
    Output('price-graph5', 'figure'),
    Output('price-graph6', 'figure')],
    [Input('town-dropdown', 'value'), 
    Input('floor_area-slider', 'value')]
)

def multi_output2(selected_town, floor_area):  #as you have more inputs, e.g 3, you will then have 3 variable in the brackets
    filtered_data = data[data.town == selected_town] #to filter only those that are same as the selected town
    filtered_data = data[data.floor_area_sqm < floor_area]
    
    flat_type = filtered_data['flat_type'].unique()
    average = []
    numTransaction = []
    for j in range(len(flat_type)):
        price = 0
        count = 0
        for i in range(len(filtered_data)):
            if flat_type[j] == filtered_data['flat_type'].iloc[i]:
                price = price + filtered_data['resale_price'].iloc[i]
                count = count + 1
        average.append(price/count)
        numTransaction.append(count)
    fig4 = px.bar(x=flat_type, y=average, labels={
        "x": "flat type",
        "y": "average price",
        "title": "Average price for each type of house"
        }
    )
    
    #k-NN algorithm
    train = filtered_data[:(int((len(filtered_data)*0.8)))]
    test = filtered_data[(int((len(filtered_data)*0.8))):]

    train_y = train[["resale_price"]]
    train_x = train[["floor_area_sqm", "lease_commence_date"]]
    
    test_y = test[["resale_price"]]
    test_x = test[["floor_area_sqm", "lease_commence_date"]]

    knn = KNeighborsRegressor(n_neighbors=3)
    print(knn)
    
    knn.fit(train_x, train_y)

    predictY= knn.predict(test_x)
    print(predictY)
    
    fig5 = px.scatter(x=filtered_data['lease_commence_date'], y=filtered_data['floor_area_sqm'], color=filtered_data['flat_type'], size=filtered_data['resale_price'], labels={"x": "Year built","y": "Floor space(sqm)"})
    
    #k-mean algorithm
    filtered_data = filtered_data[["floor_area_sqm", "resale_price","lease_commence_date"]]
    kmeans = KMeans(n_clusters=4, random_state=0).fit(filtered_data)
    labels = kmeans.labels_ #since there are 3 clusters, they will be labelled with 2,1 or 0. If 2 cluster, then 1 or 0.
    fig6 = px.scatter(x=filtered_data['floor_area_sqm'], y=filtered_data['resale_price'], color=labels, labels={"x": "floor space (sqm)","y": "Price"})
    
    
    return fig4, fig5, fig6

app1.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})
app2.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

@server.route('/hi')
def hello():                 
    return "Welcome to this Dashboard application for hdb housing price"

@server.route('/dashboard')
def render_dashboard():
    return flask.redirect('/dash1')

@server.route('/reports')
def render_reports():
    return flask.redirect('/dash2')


if __name__ == '__main__':
    app1.run(debug=False)
if __name__ == '__main__':
    app2.run(debug=False)
